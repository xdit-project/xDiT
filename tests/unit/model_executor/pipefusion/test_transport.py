import pytest
import torch

from xfuser.model_executor.pipefusion import PipeFusionTransport, PipeFusionWorkItem

from tests.unit.model_executor.pipefusion._helpers import Group


def test_first_stage_queues_only_feedback_receives():
    group = Group()
    transport = PipeFusionTransport(
        group,
        first_stage=True,
        num_steps=3,
        num_patches=2,
    )

    transport.queue_receives()

    assert group.queued == [("latent", 0), ("latent", 1)] * 2


def test_transport_rejects_duplicate_receive_queuing():
    transport = PipeFusionTransport(
        Group(),
        first_stage=True,
        num_steps=1,
        num_patches=1,
    )
    transport.queue_receives()

    with pytest.raises(RuntimeError, match="already been queued"):
        transport.queue_receives()


def test_transport_retains_contiguous_wire_tensor_until_drain():
    group = Group()
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=1,
        num_patches=1,
    )
    source = torch.arange(6, dtype=torch.float64).reshape(2, 3).transpose(0, 1)

    transport.send(source, patch_index=0)

    assert group.sent[0][2].dtype == torch.float32
    assert group.sent[0][2].is_contiguous()
    transport.wait_sends()
    assert group.works[0].waited
    assert not transport._send_tensors


def test_work_item_stops_prefetch_after_the_final_patch():
    final = PipeFusionWorkItem(
        step_index=1,
        patch_index=1,
        num_steps=2,
        num_patches=2,
        first_stage=False,
    )

    assert final.needs_receive
    assert not final.has_next_receive
