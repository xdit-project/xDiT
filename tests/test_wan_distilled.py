"""
Unit tests for the Wan2.2 LightX2V distilled model components.

All tests are CPU-only and have no HuggingFace or GPU dependencies.

Run with:
    pytest tests/test_wan_distilled.py -v
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers — import only the pure functions / classes under test
# ---------------------------------------------------------------------------

from xfuser.model_executor.models.runner_models.wan import (
    _remap_lightx2v_to_diffusers,
    _distilled_scheduler_sigmas,
    _DistilledWanScheduler,
)


# ---------------------------------------------------------------------------
# _remap_lightx2v_to_diffusers
# ---------------------------------------------------------------------------


class TestRemapLightx2vToDiffusers:
    """Key remapping from LightX2V state-dict naming to diffusers naming."""

    @pytest.mark.parametrize(
        "src, expected",
        [
            # self-attention projections
            ("blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight"),
            ("blocks.0.self_attn.k.weight", "blocks.0.attn1.to_k.weight"),
            ("blocks.0.self_attn.v.weight", "blocks.0.attn1.to_v.weight"),
            ("blocks.0.self_attn.o.weight", "blocks.0.attn1.to_out.0.weight"),
            ("blocks.0.self_attn.norm_q.weight", "blocks.0.attn1.norm_q.weight"),
            ("blocks.0.self_attn.norm_k.weight", "blocks.0.attn1.norm_k.weight"),
            # cross-attention projections
            ("blocks.5.cross_attn.q.weight", "blocks.5.attn2.to_q.weight"),
            ("blocks.5.cross_attn.k.weight", "blocks.5.attn2.to_k.weight"),
            ("blocks.5.cross_attn.v.weight", "blocks.5.attn2.to_v.weight"),
            ("blocks.5.cross_attn.o.weight", "blocks.5.attn2.to_out.0.weight"),
            ("blocks.5.cross_attn.norm_q.weight", "blocks.5.attn2.norm_q.weight"),
            ("blocks.5.cross_attn.norm_k.weight", "blocks.5.attn2.norm_k.weight"),
            # feed-forward
            ("blocks.10.ffn.0.weight", "blocks.10.ffn.net.0.proj.weight"),
            ("blocks.10.ffn.2.weight", "blocks.10.ffn.net.2.weight"),
            # norms and modulation
            ("blocks.3.norm3.weight", "blocks.3.norm2.weight"),
            ("blocks.3.modulation", "blocks.3.scale_shift_table"),
            # output head
            ("head.head.weight", "proj_out.weight"),
            ("head.modulation", "scale_shift_table"),
            # text / time embeddings
            ("text_embedding.0.weight", "condition_embedder.text_embedder.linear_1.weight"),
            ("text_embedding.2.weight", "condition_embedder.text_embedder.linear_2.weight"),
            ("time_embedding.0.weight", "condition_embedder.time_embedder.linear_1.weight"),
            ("time_embedding.2.weight", "condition_embedder.time_embedder.linear_2.weight"),
            ("time_projection.1.weight", "condition_embedder.time_proj.weight"),
            # keys that should pass through unchanged (diffusers native)
            ("patch_embedding.weight", "patch_embedding.weight"),
        ],
    )
    def test_known_mappings(self, src, expected):
        assert _remap_lightx2v_to_diffusers(src) == expected

    def test_deep_block_index(self):
        # Make sure multi-digit block indices work
        result = _remap_lightx2v_to_diffusers("blocks.39.self_attn.q.weight")
        assert result == "blocks.39.attn1.to_q.weight"

    def test_modulation_not_confused_with_norm(self):
        # blocks.N.modulation → scale_shift_table, not touching norm3
        result = _remap_lightx2v_to_diffusers("blocks.7.modulation")
        assert result == "blocks.7.scale_shift_table"
        # norm3 rename should not touch modulation
        result2 = _remap_lightx2v_to_diffusers("blocks.7.norm3.weight")
        assert result2 == "blocks.7.norm2.weight"


# ---------------------------------------------------------------------------
# _distilled_scheduler_sigmas
# ---------------------------------------------------------------------------


class TestDistilledSchedulerSigmas:
    def test_shape(self):
        sigmas = _distilled_scheduler_sigmas()
        assert sigmas.shape == (5,), "Expected 4 steps + terminal 0"

    def test_terminal_zero(self):
        sigmas = _distilled_scheduler_sigmas()
        assert sigmas[-1].item() == pytest.approx(0.0)

    def test_first_sigma_is_one(self):
        sigmas = _distilled_scheduler_sigmas()
        assert sigmas[0].item() == pytest.approx(1.0)

    def test_strictly_decreasing(self):
        sigmas = _distilled_scheduler_sigmas()
        diffs = sigmas[:-1] - sigmas[1:]  # exclude terminal 0 from diff
        assert (diffs > 0).all(), "Sigmas should be strictly decreasing"

    def test_expected_values(self):
        # Verified against LightX2V scheduler with sample_shift=5,
        # denoising_step_list=[1000, 750, 500, 250]
        sigmas = _distilled_scheduler_sigmas()
        expected_timesteps = [1000.0, 937.5, 833.333, 625.0]
        for i, t in enumerate(expected_timesteps):
            assert sigmas[i].item() * 1000 == pytest.approx(t, rel=1e-3)


# ---------------------------------------------------------------------------
# _DistilledWanScheduler
# ---------------------------------------------------------------------------


class TestDistilledWanScheduler:
    @pytest.fixture
    def scheduler(self):
        return _DistilledWanScheduler()

    def test_set_timesteps_produces_four_steps(self, scheduler):
        scheduler.set_timesteps(4)
        assert len(scheduler.timesteps) == 4

    def test_num_inference_steps_is_ignored_by_schedule(self, scheduler):
        # Passing any integer should still yield the same fixed 4 timesteps
        scheduler.set_timesteps(4)
        ts_4 = scheduler.timesteps.clone()
        scheduler.set_timesteps(20)
        ts_20 = scheduler.timesteps.clone()
        assert torch.allclose(ts_4, ts_20), (
            "Schedule must be fixed regardless of num_inference_steps"
        )

    def test_timesteps_match_expected(self, scheduler):
        scheduler.set_timesteps(4)
        expected = [1000.0, 937.5, 833.333, 625.0]
        for i, t in enumerate(expected):
            assert scheduler.timesteps[i].item() == pytest.approx(t, rel=1e-3)

    def test_step_index_reset(self, scheduler):
        scheduler.set_timesteps(4)
        assert scheduler._step_index is None
        assert scheduler._begin_index is None

    def test_device_placement(self, scheduler):
        scheduler.set_timesteps(4, device="cpu")
        assert scheduler.sigmas.device.type == "cpu"
        assert scheduler.timesteps.device.type == "cpu"

    def test_uses_config_num_train_timesteps(self, scheduler):
        # Timesteps should be sigmas * config.num_train_timesteps (default 1000)
        scheduler.set_timesteps(4)
        sigmas = _distilled_scheduler_sigmas()
        expected = sigmas[:-1] * scheduler.config.num_train_timesteps
        assert torch.allclose(scheduler.timesteps.cpu(), expected.cpu())


# ---------------------------------------------------------------------------
# _validate_args (via xFuserWan22DistilledI2VModel)
# ---------------------------------------------------------------------------


class TestValidateArgs:
    """Test _validate_args without instantiating the full model."""

    @pytest.fixture
    def call_validate(self):
        """Return a callable that invokes _validate_args with a controlled config."""
        from xfuser.model_executor.models.runner_models.wan import (
            xFuserWan22DistilledI2VModel,
        )
        from unittest.mock import MagicMock, patch

        # Minimal real subclass so super() works; __init__ bypassed entirely.
        class _TestModel(xFuserWan22DistilledI2VModel):
            def __init__(self):
                pass

        def _call(args, *, transformer_path="/fake/high.safetensors", transformer_2_path="/fake/low.safetensors"):
            instance = _TestModel()
            instance.config = MagicMock()
            instance.config.distilled_transformer_path = transformer_path
            instance.config.distilled_transformer_2_path = transformer_2_path
            # Suppress the parent chain (requires distributed init)
            with patch.object(
                xFuserWan22DistilledI2VModel.__bases__[0],
                "_validate_args",
                lambda self, a: None,
            ):
                xFuserWan22DistilledI2VModel._validate_args(instance, args)

        return _call

    def test_accepts_four_steps(self, call_validate):
        call_validate({"num_inference_steps": 4, "input_images": []})

    def test_rejects_wrong_steps(self, call_validate):
        with pytest.raises(ValueError, match="num_inference_steps must be 4"):
            call_validate({"num_inference_steps": 20, "input_images": []})

    def test_rejects_missing_transformer_path(self, call_validate):
        with pytest.raises(ValueError, match="distilled_transformer_path"):
            call_validate({"num_inference_steps": 4, "input_images": []}, transformer_path=None)

    def test_rejects_missing_transformer_2_path(self, call_validate):
        with pytest.raises(ValueError, match="distilled_transformer_2_path"):
            call_validate({"num_inference_steps": 4, "input_images": []}, transformer_2_path=None)
