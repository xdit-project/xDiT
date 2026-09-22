from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xfuser.core.distributed.attention_backend import (
    AttentionBackendType,
    FP8_HADAMARD_MATRIX,
    rotate_qk_for_fp8_comms,
)
from xfuser.core.distributed.attention_schedule import AttentionSchedule
from xfuser.core.distributed.fp8_comms import (
    Fp8CommsCall,
    Fp8CommsState,
    fp8_attention_kwargs,
    register_fp8_comms_eligible_modules,
    validate_fp8_comms_config,
)
from xfuser.model_executor.models.transformers.transformer_sd3 import (
    sd3_attn_modules,
)
from xfuser.model_executor.models.transformers.transformer_z_image import (
    z_image_attn_modules,
)

_HB_DEVICE = next(iter(FP8_HADAMARD_MATRIX))


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = nn.Module()


class _FakeTransformer(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(num_layers)])
        for i, block in enumerate(self.blocks):
            for name in ("fp8_q_scale", "fp8_k_scale", "fp8_v_scale", "fp8_o_scale"):
                block.attn1.register_buffer(name, torch.ones(1, dtype=torch.float32))
            block.attn1.register_buffer(
                "fp8_comms_layer_idx", torch.tensor([i], dtype=torch.long)
            )
        register_fp8_comms_eligible_modules(
            self, [block.attn1 for block in self.blocks]
        )


def _attach_fp8_owner(model):
    for block in model.blocks:
        object.__setattr__(block.attn1, "fp8_comms_owner", model)


def test_per_layer_running_max_and_scatter():
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=2)
    fp8.register_model(model, num_layers=2)

    layer_idx = torch.tensor([0], dtype=torch.long)
    q = torch.tensor([[[[2.0, -1.0]]]])
    k = torch.tensor([[[[4.0]]]])
    v = torch.tensor([[[[0.5]]]])
    fp8.update_running_max(model, layer_idx, q, k, v)

    layer_idx_1 = torch.tensor([1], dtype=torch.long)
    fp8.update_running_max(model, layer_idx_1, q * 3, k * 2, v * 4)

    model_state = fp8.get_model_state(model)
    assert model_state.q_running_max[0].item() == 2.0
    assert model_state.q_running_max[1].item() == 6.0
    assert model_state.k_running_max[1].item() == 8.0
    assert model_state.v_running_max[1].item() == 2.0

    fp8._scatter_scales_to_model(
        model,
        model_state.q_running_max / 100.0,
        model_state.k_running_max / 100.0,
        model_state.v_running_max / 100.0,
        model_state.o_running_max / 100.0,
    )

    assert model.blocks[0].attn1.fp8_q_scale.item() == pytest.approx(0.02)
    assert model.blocks[1].attn1.fp8_k_scale.item() == pytest.approx(0.08)


def test_fixed_scale_broadcast():
    fp8 = Fp8CommsState(fixed_scale=0.5)
    model = _FakeTransformer(num_layers=2)
    fp8.register_model(model, num_layers=2)

    assert fp8.get_model_state(model).synced is True
    assert model.blocks[0].attn1.fp8_q_scale.item() == 0.5
    assert model.blocks[1].attn1.fp8_v_scale.item() == 0.5
    assert model.blocks[1].attn1.fp8_o_scale.item() == 0.5


def test_unexercised_model_has_zero_running_max():
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)

    model_state = fp8.get_model_state(model)
    assert model_state.synced is False
    assert model_state.q_running_max.max() == 0


def test_fp8_comms_rejects_pipefusion():
    config = SimpleNamespace(
        use_fp8_comms=True,
        pipefusion_parallel_degree=2,
    )
    capabilities = SimpleNamespace(use_fp8_comms=True)
    settings = SimpleNamespace(model_name="test-model")

    with pytest.raises(ValueError, match="does not support PipeFusion"):
        validate_fp8_comms_config(config, capabilities, settings)


def test_sd3_registration_includes_dual_attention_layers():
    attn, attn2, final_attn = nn.Module(), nn.Module(), nn.Module()
    transformer = SimpleNamespace(
        transformer_blocks=[
            SimpleNamespace(attn=attn, attn2=attn2),
            SimpleNamespace(attn=final_attn, attn2=None),
        ]
    )

    assert sd3_attn_modules(transformer) == [attn, attn2, final_attn]


def test_z_image_registration_includes_refiners():
    noise, context, main = nn.Module(), nn.Module(), nn.Module()
    transformer = SimpleNamespace(
        noise_refiner=[SimpleNamespace(attention=noise)],
        context_refiner=[SimpleNamespace(attention=context)],
        layers=[SimpleNamespace(attention=main)],
    )

    assert z_image_attn_modules(transformer) == [noise, context, main]


def _outlier_qk(head_dim: int = 128, seq: int = 64, dtype=torch.bfloat16, device=_HB_DEVICE):
    """Q/K with a per-channel outlier, the distribution the rotation exists to fix.

    Seeded on CPU for determinism, then moved to ``device`` so the rotation can
    index FP8_HADAMARD_MATRIX (which only has the host's real device as a key).
    """
    torch.manual_seed(0)
    q = torch.randn(1, seq, 2, head_dim, dtype=dtype)
    q[..., 7 % head_dim] *= 30
    k = torch.randn(1, seq, 2, head_dim, dtype=dtype)
    k[..., 19 % head_dim] *= 20
    return q.to(device), k.to(device)


def test_rotation_is_qk_preserving():
    # Tolerances reflect the matrix being stored in bf16: rounding 1/sqrt(128) leaves
    # R @ R.T off the identity by ~2e-4, so scores move by ~5e-4 relative.
    q, k = _outlier_qk(dtype=torch.float32)
    q_rot, k_rot = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.AITER_FP8)

    scores = torch.matmul(q, k.transpose(-1, -2))
    scores_rot = torch.matmul(q_rot, k_rot.transpose(-1, -2))
    torch.testing.assert_close(scores, scores_rot, rtol=5e-3, atol=5e-2)


def test_rotation_shrinks_outlier_amax():
    # The rotation only helps because it moves amax, which is exactly why the
    # calibration has to measure the rotated tensor and not the input.
    q, k = _outlier_qk()
    q_rot, k_rot = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.AITER_FP8)
    assert q_rot.abs().amax() < q.abs().amax()
    assert k_rot.abs().amax() < k.abs().amax()


def test_rotation_is_a_noop_for_backends_that_do_not_rotate():
    # AITER_FP8 and AITER_FLYDSL_FP8 rotate; any other backend returns the
    # inputs untouched, so the head_dim never reaches the rotation and device
    # does not matter here.
    q, k = _outlier_qk(head_dim=8, seq=4, device="cpu")
    q_out, k_out = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.SDPA)
    assert q_out is q and k_out is k


def test_flydsl_fp8_rotates_like_aiter_fp8():
    q, k = _outlier_qk()
    q_aiter, k_aiter = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.AITER_FP8)
    q_fly, k_fly = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.AITER_FLYDSL_FP8)
    torch.testing.assert_close(q_fly, q_aiter)
    torch.testing.assert_close(k_fly, k_aiter)
    assert q_fly.abs().amax() < q.abs().amax()


def test_calibration_that_measures_nothing_raises():
    # Falling back to bf16 here would make --use_fp8_comms a silent no-op, so a
    # calibration pass that observed no activations is an error, not a downgrade.
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)
    _attach_fp8_owner(model)

    with pytest.raises(RuntimeError, match="observed no activations"):
        fp8.sync(model)
    assert fp8.get_model_state(model).synced is False


def test_observe_output_records_only_during_calibration():
    fp8 = Fp8CommsState()
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)
    _attach_fp8_owner(model)
    attn = model.blocks[0].attn1

    out = torch.tensor([[[[3.0, -4.0]]]])
    fp8.observe_output(attn, out)
    assert fp8.get_model_state(model).o_running_max[0].item() == 4.0

    fp8.get_model_state(model).synced = True
    fp8.observe_output(attn, out * 10)
    assert fp8.get_model_state(model).o_running_max[0].item() == 4.0


def test_fp8_attention_kwargs_returns_call_when_synced():
    fp8 = Fp8CommsState(fixed_scale=0.5)
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)
    _attach_fp8_owner(model)
    attn = model.blocks[0].attn1
    q = torch.ones(1, 2, 2, 4)

    extras = fp8_attention_kwargs(
        fp8, attn, q, q, q, False, AttentionBackendType.AITER_FLYDSL_FP8
    )
    assert "fp8_comms" in extras
    assert isinstance(extras["fp8_comms"], Fp8CommsCall)
    assert extras["fp8_comms"].q_scale.item() == 0.5


def test_hybrid_schedule_gates_fp8_comms_by_active_backend():
    fp8 = Fp8CommsState(fixed_scale=0.5)
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)
    _attach_fp8_owner(model)
    attn = model.blocks[0].attn1
    q = torch.ones(1, 2, 2, 4)
    schedule = AttentionSchedule(
        [
            AttentionBackendType.AITER_FLYDSL_FP8,
            AttentionBackendType.SDPA,
            AttentionBackendType.AITER_FP8,
        ]
    )

    extras_by_step = [
        fp8_attention_kwargs(
            fp8, attn, q, q, q, False, schedule.get_backend(step)
        )
        for step in range(schedule.total_steps)
    ]

    assert isinstance(extras_by_step[0]["fp8_comms"], Fp8CommsCall)
    assert extras_by_step[1] == {}
    assert isinstance(extras_by_step[2]["fp8_comms"], Fp8CommsCall)


def test_hadamard_matrix_is_orthonormal():
    R = FP8_HADAMARD_MATRIX[_HB_DEVICE].float()
    torch.testing.assert_close(
        R @ R.T, torch.eye(R.shape[0], device=R.device), rtol=0, atol=1e-3
    )


def test_calibrated_scale_matches_the_rotated_tensor():
    """The frozen scale must describe the tensor that is actually quantized.

    Calibrating on unrotated amaxes leaves the scale too small for the rotated
    tensor whenever rotation raises amax, which clips; ``safety_factor`` alone only
    covers 1/0.85 of that.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    safety_factor = 0.85
    fp8 = Fp8CommsState(safety_factor=safety_factor)
    model = _FakeTransformer(num_layers=1)
    fp8.register_model(model, num_layers=1)
    # Running-max buffers default to CPU; move them to the rotation device so the
    # in-place maximum in update_running_max stays on one device.
    fp8.get_model_state(model).to_device_(_HB_DEVICE)

    q, k = _outlier_qk()
    v = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16, device=_HB_DEVICE)
    q_rot, k_rot = rotate_qk_for_fp8_comms(q, k, AttentionBackendType.AITER_FP8)

    fp8.update_running_max(
        model, torch.tensor([0], dtype=torch.long, device=_HB_DEVICE), q_rot, k_rot, v
    )
    model_state = fp8.get_model_state(model)

    assert model_state.q_running_max[0].item() == pytest.approx(
        q_rot.abs().amax().float().item()
    )
    assert model_state.q_running_max[0].item() != pytest.approx(
        q.abs().amax().float().item()
    )

    scales = torch.stack(
        [
            model_state.q_running_max,
            model_state.k_running_max,
            model_state.v_running_max,
            model_state.o_running_max,
        ]
    ).clamp(min=1e-6) / (fp8_max * safety_factor)
    fp8._scatter_scales_to_model(model, scales[0], scales[1], scales[2], scales[3])

    # Quantizing the rotated tensor with the frozen scale must stay inside the
    # representable range, with the headroom safety_factor promises.
    q_scale = model.blocks[0].attn1.fp8_q_scale.item()
    assert (q_rot.float() / q_scale).abs().amax().item() == pytest.approx(
        fp8_max * safety_factor, rel=1e-2
    )
