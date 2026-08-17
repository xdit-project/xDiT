"""xDiT orchestration at the DistVAE boundary.

DistVAE owns adapter selection, tile planning, and tile distribution algorithms. These tests
cover only the xDiT policy that discovers VAEs, selects its runtime group, forwards CLI settings,
and presents decode failures.
"""

from types import SimpleNamespace
from unittest import mock

import diffusers
import pytest
import torch
import torch.nn as nn

from distvae.utils import ParallelContext
from distvae.vae import VAERowSplitError, parallel
from xfuser import envs
from xfuser.config import FlexibleArgumentParser, xFuserArgs
from xfuser.model_executor.pipelines import base_pipeline
from xfuser.model_executor.models.runner_models import base_model, vae_manager


class _TestRunner(base_model.xFuserModel):
    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _distributed_log_environment(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.fixture(scope="module", autouse=True)
def _torch_group_norm_for_distvae_tests():
    original = torch.nn.GroupNorm
    envs.restore_torch_group_norm_for_distvae()
    yield
    torch.nn.GroupNorm = original


def _runner(**config):
    runner = object.__new__(_TestRunner)
    defaults = {
        "enable_slicing": False,
        "enable_tiling": False,
        "vae_tile_size_height": None,
        "vae_tile_size_width": None,
        "vae_tile_overlap_height": None,
        "vae_tile_overlap_width": None,
        "enable_sequential_cpu_offload": False,
        "enable_model_cpu_offload": False,
        "use_parallel_vae": False,
    }
    defaults.update(config)
    runner.config = SimpleNamespace(**defaults)
    runner.capabilities = base_model.ModelCapabilities(
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    runner.settings = SimpleNamespace(model_name="test-model", valid_tasks=[])
    runner._vae_manager = vae_manager.VAEManager(
        runner.config, runner.capabilities, runner.settings
    )
    return runner


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
def test_cli_propagates_vae_settings(add_args):
    parser = add_args(FlexibleArgumentParser(description="xDiT"))
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--use-parallel-vae",
            "--enable_tiling",
            "--vae_tile_size_height",
            "320",
            "--vae_tile_size_width",
            "512",
            "--vae_tile_overlap_height",
            "32",
            "--vae_tile_overlap_width",
            "64",
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.use_parallel_vae is True
    assert config.enable_tiling is True
    assert config.vae_tile_size_height == 320
    assert config.vae_tile_size_width == 512
    assert config.vae_tile_overlap_height == 32
    assert config.vae_tile_overlap_width == 64


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
def test_vae_tile_settings_document_tiling_requirement(add_args):
    parser = add_args(FlexibleArgumentParser(description="xDiT"))
    tile_settings = [
        action
        for action in parser._actions
        if action.dest.startswith("vae_tile_")
    ]

    assert len(tile_settings) == 4
    assert all("Requires --enable_tiling." in action.help for action in tile_settings)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"vae_tile_size_height": 320},
            "--vae_tile_size_height and --vae_tile_size_width must be provided together",
        ),
        (
            {"vae_tile_size_width": 512},
            "--vae_tile_size_height and --vae_tile_size_width must be provided together",
        ),
        (
            {"vae_tile_size_height": 0, "vae_tile_size_width": 512},
            "--vae_tile_size_height must be positive",
        ),
        (
            {"vae_tile_size_height": 320, "vae_tile_size_width": -1},
            "--vae_tile_size_width must be positive",
        ),
    ],
)
def test_rectangular_vae_tile_settings_are_validated(config, message):
    runner = _runner()

    with pytest.raises(ValueError, match=message):
        vae_manager.validate_vae_config(
            xFuserArgs(model="test-model", **config),
            runner.capabilities,
            runner.settings,
        )


@pytest.mark.parametrize(
    "config",
    [
        {"vae_tile_size_height": 320, "vae_tile_size_width": 512},
        {"vae_tile_overlap_height": 32, "vae_tile_overlap_width": 64},
    ],
)
def test_vae_tile_settings_require_tiling(config):
    runner = _runner()

    with pytest.raises(ValueError, match="require --enable_tiling"):
        vae_manager.validate_vae_config(
            xFuserArgs(model="test-model", **config),
            runner.capabilities,
            runner.settings,
        )


def test_rectangular_vae_tile_flag_does_not_enable_tiling():
    runner = _runner(vae_tile_size_height=320, vae_tile_size_width=None)

    assert runner._vae_manager._tiling_flag() is None


def test_tile_overlap_flag_does_not_enable_tiling():
    runner = _runner(vae_tile_overlap_height=32, vae_tile_overlap_width=64)

    assert runner._vae_manager._tiling_flag() is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"vae_tile_overlap_height": 32},
            "--vae_tile_overlap_height and --vae_tile_overlap_width must be provided together",
        ),
        (
            {"vae_tile_overlap_width": 64},
            "--vae_tile_overlap_height and --vae_tile_overlap_width must be provided together",
        ),
        (
            {"vae_tile_overlap_height": -1, "vae_tile_overlap_width": 64},
            "--vae_tile_overlap_height must be non-negative",
        ),
        (
            {"vae_tile_overlap_height": 32, "vae_tile_overlap_width": -1},
            "--vae_tile_overlap_width must be non-negative",
        ),
    ],
)
def test_per_axis_vae_tile_overlap_settings_are_validated(config, message):
    runner = _runner()

    with pytest.raises(ValueError, match=message):
        vae_manager.validate_vae_config(
            xFuserArgs(model="test-model", **config),
            runner.capabilities,
            runner.settings,
        )


def test_zero_overlap_is_valid_for_an_inactive_strip_axis():
    runner = _runner()
    config = xFuserArgs(
        model="test-model",
        enable_tiling=True,
        vae_tile_overlap_height=0,
        vae_tile_overlap_width=64,
    )

    vae_manager.validate_vae_config(
        config, runner.capabilities, runner.settings
    )


def test_vae_validator_restores_aiter_groupnorm_for_parallel_decode():
    runner = _runner()
    config = xFuserArgs(model="test-model", use_parallel_vae=True)

    with mock.patch.object(
        vae_manager,
        "restore_torch_group_norm_for_distvae",
        return_value=True,
    ) as restore:
        vae_manager.validate_vae_config(
            config,
            runner.capabilities,
            runner.settings,
        )

    restore.assert_called_once_with()


def test_enable_tiling_without_dimensions_keeps_native_window():
    vae = SimpleNamespace(enable_tiling=mock.Mock(), decode=mock.Mock())
    runner = _runner(enable_tiling=True)
    runner.pipe = SimpleNamespace(vae=vae)

    with (
        mock.patch.object(vae_manager.vae_tiling, "require_vae_support"),
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape", return_value=(512, 512)
        ),
        mock.patch.object(vae_manager.vae_tiling, "apply_tile_plan") as apply,
        mock.patch.object(runner._vae_manager, "_apply_vae_tile_overlap"),
        mock.patch.object(
            runner._vae_manager, "_check_tiles_against_parallel_vae"
        ),
        mock.patch.object(runner._vae_manager, "_install_vae_tiled_decode"),
        mock.patch.object(runner._vae_manager, "_install_vae_decode_guard"),
    ):
        runner._vae_manager.enable_options([vae])

    vae.enable_tiling.assert_called_once_with()
    apply.assert_not_called()


def test_worker_vae_wrapper_uses_runtime_device_group_with_public_api():
    vae = SimpleNamespace()
    device_group = object()
    coordinator = SimpleNamespace(device_group=device_group)

    with (
        mock.patch.object(
            base_pipeline, "get_vae_parallel_group", return_value=coordinator
        ),
        mock.patch.object(
            base_pipeline, "parallelize_decoder"
        ) as parallelize_decoder,
    ):
        converted = base_pipeline.xFuserVAEWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, device_group)


def test_worker_vae_wrapper_accepts_dedicated_raw_process_group():
    vae = SimpleNamespace()
    process_group = object()

    with (
        mock.patch.object(
            base_pipeline, "get_vae_parallel_group", return_value=process_group
        ),
        mock.patch.object(
            base_pipeline, "parallelize_decoder"
        ) as parallelize_decoder,
    ):
        converted = base_pipeline.xFuserVAEWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, process_group)


def test_pipeline_vae_conversion_uses_runtime_vae_group():
    vae = SimpleNamespace()
    device_group = object()
    coordinator = SimpleNamespace(device_group=device_group)

    with (
        mock.patch.object(
            base_pipeline, "get_vae_parallel_group", return_value=coordinator
        ),
        mock.patch.object(
            base_pipeline, "parallelize_decoder"
        ) as parallelize_decoder,
    ):
        converted = base_pipeline.xFuserPipelineBaseWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, device_group)


def test_parallel_setup_adapts_runtime_group_for_distvae():
    tiled, sharded = SimpleNamespace(use_tiling=True), SimpleNamespace()
    device_group = object()
    coordinator = SimpleNamespace(
        device_group=device_group,
        rank_in_group=1,
        world_size=3,
        ranks=[4, 7, 9],
    )
    runner = _runner(use_parallel_vae=True)
    runner.pipe = SimpleNamespace(vae=tiled)
    runner.second_pipe = SimpleNamespace(vae=sharded)

    with (
        mock.patch.object(
            vae_manager, "get_vae_parallel_group", return_value=coordinator
        ),
        mock.patch.object(torch.distributed, "get_world_size", return_value=3),
        mock.patch.object(torch.distributed, "get_rank", return_value=1),
        mock.patch.object(
            vae_manager.vae_tiling,
            "supports_tile_parallel",
            side_effect=[True, False],
        ),
        mock.patch.object(vae_manager, "mark") as mark,
        mock.patch.object(
            vae_manager.vae_parallel,
            "parallelize_decoder",
            return_value="DecoderAdapter",
        ) as parallelize_decoder,
        mock.patch.object(
            vae_manager.vae_parallel,
            "parallelize_encoder",
            return_value="EncoderAdapter",
        ) as parallelize_encoder,
    ):
        runner._vae_manager.setup_parallel_vae([tiled, sharded])

    context = mark.call_args.args[1]
    assert mark.call_args.args[0] is tiled
    assert context == ParallelContext(
        group=device_group,
        rank=1,
        world_size=3,
        patch_dim=-2,
        global_ranks=(4, 7, 9),
    )
    parallelize_decoder.assert_called_once_with(sharded, device_group)
    assert parallelize_encoder.call_args_list == [
        mock.call(tiled, device_group),
        mock.call(sharded, device_group),
    ]


def test_tile_overlap_is_applied_through_distvae():
    vae = SimpleNamespace()
    runner = _runner(
        vae_tile_overlap_height=32,
        vae_tile_overlap_width=64,
        height=320,
        width=1280,
    )
    overlap_plan = {
        "tile_overlap_factor_height": 0.0625,
        "tile_overlap_factor_width": 0.125,
    }

    with (
        mock.patch.object(
            vae_manager.vae_tiling,
            "tile_overlap",
            side_effect=[(128, 128), (32, 64)],
        ),
        mock.patch.object(
            vae_manager.vae_tiling,
            "tile_overlap_plan",
            return_value=overlap_plan,
        ) as tile_overlap_plan,
        mock.patch.object(vae_manager.vae_tiling, "apply_tile_plan") as apply,
    ):
        runner._vae_manager._apply_vae_tile_overlap(vae, (320, 1280))

    tile_overlap_plan.assert_called_once_with(
        vae, 32, 64, sample_shape=(320, 1280)
    )
    apply.assert_called_once_with(vae, overlap_plan)


def test_tile_overlap_tracks_each_run_sample_shape():
    vae = SimpleNamespace()
    runner = _runner(
        vae_tile_overlap_height=32,
        vae_tile_overlap_width=64,
    )

    with mock.patch.object(
        runner._vae_manager, "_apply_vae_tile_overlap"
    ) as apply:
        runner._vae_manager.prepare_run(
            [vae], {"height": 320, "width": 1280}
        )
        runner._vae_manager.prepare_run(
            [vae], {"height": 320, "width": 1280}
        )
        runner._vae_manager.prepare_run(
            [vae], {"height": 640, "width": 960}
        )

    assert apply.call_args_list == [
        mock.call(vae, (320, 1280)),
        mock.call(vae, (640, 960)),
    ]


def test_tile_overlap_refuses_an_inexact_plan_without_mutating_the_vae():
    vae = SimpleNamespace()
    runner = _runner(
        vae_tile_overlap_height=33,
        vae_tile_overlap_width=65,
        height=1024,
        width=1024,
    )

    with (
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape", return_value=(512, 768)
        ),
        mock.patch.object(
            vae_manager.vae_tiling,
            "tile_overlap_plan",
            return_value=None,
        ),
        mock.patch.object(vae_manager.vae_tiling, "apply_tile_plan") as apply,
    ):
        with pytest.raises(ValueError) as error:
            runner._vae_manager._apply_vae_tile_overlap(vae, (1024, 1024))

    assert "33x65 pixels" in str(error.value)
    assert "512x768" in str(error.value)
    apply.assert_not_called()


def test_rectangular_tile_shape_is_applied_exactly_through_distvae():
    vae = SimpleNamespace()
    runner = _runner(vae_tile_size_height=320, vae_tile_size_width=512)
    plan = {
        "tile_sample_min_height": 320,
        "tile_sample_min_width": 512,
        "tile_latent_min_height": 40,
        "tile_latent_min_width": 64,
    }

    with (
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape_plan", return_value=plan
        ) as tile_shape_plan,
        mock.patch.object(vae_manager.vae_tiling, "apply_tile_plan") as apply,
    ):
        assert runner._vae_manager._apply_vae_tile_shape(vae) == (320, 512)

    tile_shape_plan.assert_called_once_with(vae, 320, 512)
    apply.assert_called_once_with(vae, plan)


def test_rectangular_tile_shape_refuses_a_non_exact_plan_actionably():
    vae = SimpleNamespace()
    runner = _runner(vae_tile_size_height=321, vae_tile_size_width=512)

    with mock.patch.object(
        vae_manager.vae_tiling, "tile_shape_plan", return_value=None
    ):
        with pytest.raises(ValueError) as error:
            runner._vae_manager._apply_vae_tile_shape(vae)

    message = str(error.value)
    assert "--vae_tile_size_height 321" in message
    assert "--vae_tile_size_width 512" in message
    assert "exactly" in message


def test_rectangular_tile_shape_is_applied_to_every_staged_vae():
    first = SimpleNamespace(enable_tiling=mock.Mock(), decode=mock.Mock())
    second = SimpleNamespace(enable_tiling=mock.Mock(), decode=mock.Mock())
    runner = _runner(
        enable_tiling=True,
        vae_tile_size_height=320,
        vae_tile_size_width=512,
    )
    runner.pipe = SimpleNamespace(vae=first)
    runner.second_pipe = SimpleNamespace(vae=second)

    with (
        mock.patch.object(vae_manager.vae_tiling, "require_vae_support"),
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape", return_value=(512, 512)
        ),
        mock.patch.object(
            runner._vae_manager,
            "_apply_vae_tile_shape",
            return_value=(320, 512),
        ) as apply_shape,
        mock.patch.object(runner._vae_manager, "_apply_vae_tile_overlap"),
        mock.patch.object(
            runner._vae_manager, "_check_tiles_against_parallel_vae"
        ),
        mock.patch.object(runner._vae_manager, "_install_vae_tiled_decode"),
        mock.patch.object(runner._vae_manager, "_install_vae_decode_guard"),
    ):
        runner._vae_manager.enable_options([first, second])

    assert apply_shape.call_args_list == [mock.call(first), mock.call(second)]


def test_single_gpu_rectangular_scalar_vae_installs_undispatched_tiled_decode():
    vae = SimpleNamespace()
    runner = _runner(
        use_parallel_vae=False,
        vae_tile_size_height=320,
        vae_tile_size_width=512,
    )
    installed = mock.Mock()

    with (
        mock.patch.object(
            vae_manager.vae_tile_parallel, "context_of", return_value=None
        ),
        mock.patch.object(
            vae_manager.vae_tiling, "tiled_decode_for", return_value=installed
        ) as tiled_decode_for,
    ):
        runner._vae_manager._install_vae_tiled_decode(vae)

    tiled_decode_for.assert_called_once_with(vae)
    assert vae.tiled_decode is installed


def test_native_keyed_vae_keeps_its_tiled_decode_when_distvae_returns_none():
    native = mock.Mock()
    vae = SimpleNamespace(tiled_decode=native)
    runner = _runner(vae_tile_size_height=320, vae_tile_size_width=512)

    with (
        mock.patch.object(
            vae_manager.vae_tile_parallel, "context_of", return_value=None
        ),
        mock.patch.object(
            vae_manager.vae_tiling, "tiled_decode_for", return_value=None
        ) as tiled_decode_for,
    ):
        runner._vae_manager._install_vae_tiled_decode(vae)

    tiled_decode_for.assert_called_once_with(vae)
    assert vae.tiled_decode is native


def test_tiled_decode_install_uses_distvae_context_and_sharing():
    group = object()
    context = ParallelContext(
        group=group,
        rank=0,
        world_size=2,
        patch_dim=-2,
        global_ranks=(0, 1),
    )
    vae = SimpleNamespace()
    runner = _runner()
    installed = mock.Mock()

    with (
        mock.patch.object(
            vae_manager.vae_tile_parallel, "context_of", return_value=context
        ),
        mock.patch.object(
            vae_manager.vae_tile_parallel,
            "sharing",
            return_value=("dispatch", "assemble"),
        ) as sharing,
        mock.patch.object(
            vae_manager.vae_tiling,
            "tiled_decode_for",
            return_value=installed,
        ) as tiled_decode_for,
    ):
        runner._vae_manager._install_vae_tiled_decode(vae)

    sharing.assert_called_once_with(context)
    tiled_decode_for.assert_called_once_with(vae, "dispatch", "assemble")
    assert vae.tiled_decode is installed


def test_parallel_vae_rank_hint_increases_only_the_sharded_tile_axis():
    vae = SimpleNamespace()
    runner = _runner(use_parallel_vae=True)

    def shape_plan(_vae, height, width):
        if width != 64 or height % 32:
            return None
        return {"rows": height // 64}

    def rows(_vae, plan=None):
        return 1 if plan is None else plan["rows"]

    with (
        mock.patch.object(
            vae_manager.vae_tile_parallel, "context_of", return_value=None
        ),
        mock.patch.object(
            vae_manager, "get_vae_parallel_world_size", return_value=2
        ),
        mock.patch.object(vae_manager, "latent_rows", side_effect=rows),
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape", return_value=(64, 64)
        ),
        mock.patch.object(
            vae_manager.vae_tiling, "tile_shape_plan", side_effect=shape_plan
        ) as tile_shape_plan,
    ):
        with pytest.raises(ValueError) as error:
            runner._vae_manager._check_tiles_against_parallel_vae(
                vae, (512, 512)
            )

    assert "--vae_tile_size_height 128 --vae_tile_size_width 64" in str(error.value)
    assert tile_shape_plan.call_args_list[0] == mock.call(vae, 64, 64)


def test_parallel_vae_accepts_rectangular_tile_with_enough_latent_height():
    vae = SimpleNamespace(
        tile_sample_min_height=256,
        tile_sample_min_width=64,
        tile_latent_min_height=32,
        tile_latent_min_width=8,
    )
    runner = _runner(use_parallel_vae=True)

    with (
        mock.patch.object(
            vae_manager.vae_tile_parallel, "context_of", return_value=None
        ),
        mock.patch.object(
            vae_manager, "get_vae_parallel_world_size", return_value=16
        ),
    ):
        runner._vae_manager._check_tiles_against_parallel_vae(
            vae, native_shape=(512, 512)
        )


def test_decode_guard_reports_rectangular_window_and_flags():
    original = mock.Mock(side_effect=RuntimeError("decoder padding failure"))
    vae = SimpleNamespace(decode=original)
    runner = _runner(
        vae_tile_size_height=320,
        vae_tile_size_width=512,
    )

    with mock.patch.object(
        vae_manager.vae_tiling, "is_tile_padding_error", return_value=True
    ) as is_padding:
        runner._vae_manager._install_vae_decode_guard(
            vae, tile_shape=(320, 512)
        )
        with pytest.raises(RuntimeError) as error:
            vae.decode(object())

    message = str(error.value)
    assert "320x512" in message
    assert "--vae_tile_size_height" in message
    assert "--vae_tile_size_width" in message
    is_padding.assert_called_once()


def test_decode_guard_suggests_complete_tiles_for_incompatible_row_sharding():
    vae = SimpleNamespace(
        config=SimpleNamespace(scale_factor_spatial=16),
        decode=mock.Mock(side_effect=VAERowSplitError(rows=45, factor=2)),
    )
    runner = _runner(use_parallel_vae=True)

    with mock.patch.object(
        vae_manager.vae_tiling, "supports_tile_parallel", return_value=True
    ):
        runner._vae_manager._install_vae_decode_guard(vae)
        with pytest.raises(ValueError) as error:
            vae.decode(object())

    assert str(error.value) == (
        "Cannot row-shard 45 latent rows in the VAE decoder: this VAE processes rows "
        "in groups of 2, but 45 is not divisible by 2. Use an output height divisible "
        "by 32, enable --enable_tiling to distribute complete tiles instead, or disable "
        "--use_parallel_vae."
    )


def test_decode_oom_hint_reports_rectangular_window():
    vae = SimpleNamespace(use_tiling=True)
    runner = _runner()

    with mock.patch.object(
        vae_manager.vae_tiling, "tile_shape", return_value=(320, 512)
    ):
        hint = runner._vae_manager._vae_decode_oom_hint(vae)

    assert "320x512" in hint
    assert "--vae_tile_size_height" in hint
    assert "--vae_tile_size_width" in hint
    assert "no single tile window" not in hint


class AiterGroupNorm(nn.Module):
    __module__ = "aiter.ops.groupnorm"


def _two_d_vae():
    return diffusers.AutoencoderKL(
        block_out_channels=[8, 8, 16, 16],
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
    )


def test_non_aiter_group_norm_restoration_is_a_no_op():
    original = envs._TORCH_GROUPNORM
    with mock.patch.object(nn, "GroupNorm", original):
        assert envs.restore_torch_group_norm_for_distvae() is False
        assert nn.GroupNorm is original


def test_aiter_replacement_blocks_recognition_until_xdit_restores_torch_norm():
    original = envs._TORCH_GROUPNORM
    vae = _two_d_vae()
    assert parallel.decoder_adapter_name(vae) == "DecoderAdapter"

    with mock.patch.object(nn, "GroupNorm", AiterGroupNorm):
        assert parallel.decoder_adapter_name(vae) is None
        assert envs.restore_torch_group_norm_for_distvae() is True
        assert nn.GroupNorm is original
        assert parallel.decoder_adapter_name(vae) == "DecoderAdapter"
