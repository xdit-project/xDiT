"""xDiT orchestration at the DistVAE boundary.

DistVAE owns adapter selection, tile planning, and tile distribution algorithms. These tests
cover only the xDiT policy that discovers VAEs, selects its runtime group, forwards CLI settings,
and presents decode failures.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import diffusers
import pytest
import torch
import torch.nn as nn

from distvae.utils import ParallelContext
from distvae.vae import parallel, tile_parallel, tiling
from xfuser import compat as xfuser_compat
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


def test_runner_cli_propagates_vae_settings():
    parser = xFuserArgs.add_runner_args(
        FlexibleArgumentParser(description="xDiT runner")
    )
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--use-parallel-vae",
            "--vae-tile-overlap-height",
            "32",
            "--vae-tile-overlap-width",
            "64",
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.use_parallel_vae is True
    assert config.vae_tile_overlap_height == 32
    assert config.vae_tile_overlap_width == 64


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
def test_legacy_scalar_overlap_cli_is_rejected(add_args):
    parser = add_args(FlexibleArgumentParser(description="xDiT"))

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "test-model", "--vae_tile_overlap", "0.125"]
        )


def test_xfuser_args_has_no_legacy_scalar_overlap_field():
    assert not hasattr(xFuserArgs(model="test-model"), "vae_tile_overlap")


@pytest.mark.parametrize("add_args", [xFuserArgs.add_runner_args, xFuserArgs.add_cli_args])
def test_legacy_scalar_tile_cli_is_rejected(add_args):
    parser = add_args(FlexibleArgumentParser(description="xDiT"))

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "test-model", "--vae_tile_" + "size", "384"]
        )


def test_xfuser_args_has_no_legacy_scalar_tile_field():
    assert not hasattr(xFuserArgs(model="test-model"), "vae_tile_" + "size")


def test_runner_cli_propagates_rectangular_vae_tile_settings():
    parser = xFuserArgs.add_runner_args(
        FlexibleArgumentParser(description="xDiT runner")
    )
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--vae_tile_size_height",
            "320",
            "--vae_tile_size_width",
            "512",
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.vae_tile_size_height == 320
    assert config.vae_tile_size_width == 512


def test_engine_cli_propagates_rectangular_vae_tile_settings():
    parser = xFuserArgs.add_cli_args(
        FlexibleArgumentParser(description="xDiT engine")
    )
    parsed = parser.parse_args(
        [
            "--model",
            "test-model",
            "--vae_tile_size_height",
            "320",
            "--vae_tile_size_width",
            "512",
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.vae_tile_size_height == 320
    assert config.vae_tile_size_width == 512


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


def test_rectangular_vae_tile_flag_implies_tiling():
    runner = _runner(vae_tile_size_height=320, vae_tile_size_width=None)

    assert runner._vae_manager._tiling_flag() == "--vae_tile_size_height"


def test_tile_overlap_flag_implies_tiling():
    runner = _runner(vae_tile_overlap_height=32, vae_tile_overlap_width=64)

    assert runner._vae_manager._tiling_flag() == "--vae_tile_overlap_height"


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
        vae_tile_overlap_height=0,
        vae_tile_overlap_width=64,
    )

    vae_manager.validate_vae_config(
        config, runner.capabilities, runner.settings
    )


def test_vae_validator_reports_missing_distvae_for_parallel_decode():
    runner = _runner()
    config = xFuserArgs(model="test-model", use_parallel_vae=True)

    with pytest.raises(ValueError, match="DistVAE is not installed"):
        vae_manager.validate_vae_config(
            config,
            runner.capabilities,
            runner.settings,
            package_info={"has_distvae": False},
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
            package_info={"has_distvae": True},
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


def test_old_config_without_rectangular_fields_keeps_tiling_helpers_compatible():
    runner = _runner()
    del runner.config.vae_tile_size_height
    del runner.config.vae_tile_size_width

    assert runner._vae_manager._tiling_flag() is None
    assert runner._vae_manager._requested_vae_tile_shape() is None


def test_old_config_without_rectangular_fields_passes_validation():
    current = xFuserArgs(model="test-model")
    old_config = SimpleNamespace(
        **{
            name: value
            for name, value in vars(current).items()
            if name not in {"vae_tile_size_height", "vae_tile_size_width"}
        }
    )
    runner = _runner()

    vae_manager.validate_vae_config(
        old_config, runner.capabilities, runner.settings
    )


def test_base_model_uses_public_distvae_modules():
    assert vae_manager.vae_parallel is parallel
    assert vae_manager.vae_tiling is tiling
    assert vae_manager.vae_tile_parallel is tile_parallel


def test_xdit_declares_the_distvae_public_api_version_floor():
    setup = (Path(__file__).parents[2] / "setup.py").read_text()
    assert '"distvae>=0.0.0beta9"' in setup


def test_compat_loader_names_required_api_when_distvae_lacks_it(tmp_path):
    package = tmp_path / "distvae"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__version__.py").write_text('__version__ = "0.0.0beta6"\n')
    vae_package = package / "vae"
    vae_package.mkdir()
    (vae_package / "__init__.py").write_text("")
    for module in ("parallel", "tile_parallel", "tiling"):
        (vae_package / f"{module}.py").write_text("")
    repo = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo)))

    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from xfuser.compat import load_distvae_vae; load_distvae_vae()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert imported.returncode != 0, imported.stderr + imported.stdout
    error = imported.stderr + imported.stdout
    assert "DistVAE>=0.0.0beta9" in error
    assert "public distvae.vae API" in error


def test_compat_loader_rejects_beta8_even_with_every_required_symbol(tmp_path):
    package = tmp_path / "distvae"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__version__.py").write_text('__version__ = "0.0.0beta8"\n')
    vae_package = package / "vae"
    vae_package.mkdir()
    for module_name, required in xfuser_compat._DISTVAE_VAE_API.items():
        path = (
            vae_package / "__init__.py"
            if module_name == "distvae.vae"
            else vae_package / f"{module_name.rsplit('.', 1)[-1]}.py"
        )
        path.write_text("\n".join(f"{name} = object()" for name in required))
    repo = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repo)))

    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "from xfuser.compat import load_distvae_vae; load_distvae_vae()",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert imported.returncode != 0, imported.stderr + imported.stdout
    error = imported.stderr + imported.stdout
    assert "DistVAE>=0.0.0beta9" in error
    assert "Upgrade DistVAE" in error


def test_distvae_compat_loader_returns_normal_local_public_modules():
    loaded = xfuser_compat.load_distvae_vae()
    assert loaded == (parallel, tile_parallel, tiling)


def test_distvae_compat_requires_beta9_canonical_api_only():
    required = {
        name
        for names in xfuser_compat._DISTVAE_VAE_API.values()
        for name in names
    }
    removed = {
        "local_tiled_decode_for",
        "smallest_tile_window",
        "snap_tile_window",
        "tile_plan",
        "tile_window",
        "widest_tile_overlap",
    }

    assert removed.isdisjoint(required)
    assert {
        "apply_tile_plan",
        "latent_rows",
        "tile_overlap_plan",
        "tile_shape",
        "tile_shape_plan",
        "tiled_decode_for",
    } <= required


def test_all_xdit_python_callers_use_public_distvae_boundaries():
    xfuser = Path(__file__).parents[2] / "xfuser"
    removed = {
        "xfuser.core.utils.vae_parallel",
        "xfuser.core.utils.vae_tiling",
        "xfuser.core.utils.vae_tile_parallel",
    }
    removed_names = {module.rsplit(".", 1)[1] for module in removed}
    violations = []
    for path in xfuser.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.module == "xfuser.core.utils":
                    modules.extend(
                        f"{node.module}.{name.name}"
                        for name in node.names
                        if name.name in removed_names
                    )
            else:
                continue
            forbidden = [
                module
                for module in modules
                if module.startswith("distvae.modules.adapters") or module in removed
            ]
            if forbidden:
                violations.append(
                    (str(path.relative_to(xfuser)), node.lineno, forbidden)
                )

    assert violations == []


def test_worker_vae_wrapper_uses_runtime_device_group_with_public_api():
    vae = SimpleNamespace()
    device_group = object()
    coordinator = SimpleNamespace(device_group=device_group)

    with (
        mock.patch.object(
            base_pipeline, "get_vae_parallel_group", return_value=coordinator
        ),
        mock.patch.object(
            base_pipeline.vae_parallel, "parallelize_decoder"
        ) as parallelize_decoder,
    ):
        converted = base_pipeline.xFuserVAEWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, device_group)


def test_worker_vae_wrapper_accepts_a_dedicated_raw_process_group():
    vae = SimpleNamespace()
    process_group = object()

    with (
        mock.patch.object(
            base_pipeline, "get_vae_parallel_group", return_value=process_group
        ),
        mock.patch.object(
            base_pipeline.vae_parallel, "parallelize_decoder"
        ) as parallelize_decoder,
    ):
        converted = base_pipeline.xFuserVAEWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, process_group)


def test_pipeline_vae_conversion_uses_public_api_world_group_default():
    vae = SimpleNamespace()

    with mock.patch.object(
        base_pipeline.vae_parallel, "parallelize_decoder"
    ) as parallelize_decoder:
        converted = base_pipeline.xFuserPipelineBaseWrapper._convert_vae(object(), vae)

    assert converted is vae
    parallelize_decoder.assert_called_once_with(vae, None)


def test_decoding_vaes_include_unique_staged_pipeline_vaes():
    first, second = object(), object()
    runner = _runner()
    runner.pipe = SimpleNamespace(vae=first)
    runner.second_pipe = SimpleNamespace(vae=second)

    assert runner._vae_manager.decoding_vaes(
        runner.pipe, runner.second_pipe
    ) == [first, second]

    runner.second_pipe.vae = first
    assert runner._vae_manager.decoding_vaes(
        runner.pipe, runner.second_pipe
    ) == [first]


def test_base_enable_options_delegates_all_decoding_vaes_once():
    first, second = object(), object()
    runner = _runner()
    runner.pipe = SimpleNamespace(vae=first)
    runner.second_pipe = SimpleNamespace(vae=second)

    with mock.patch.object(
        runner._vae_manager, "enable_options"
    ) as enable_options:
        runner._enable_options()

    enable_options.assert_called_once_with([first, second])


def test_base_parallel_and_channels_last_methods_are_thin_delegators():
    vae = object()
    runner = _runner()
    runner.pipe = SimpleNamespace(vae=vae)

    with (
        mock.patch.object(
            runner._vae_manager, "setup_parallel_vae"
        ) as setup_parallel,
        mock.patch.object(
            runner._vae_manager, "convert_to_channels_last"
        ) as convert,
    ):
        runner._setup_parallel_vae()
        runner._convert_vae_to_channels_last()

    setup_parallel.assert_called_once_with([vae])
    convert.assert_called_once_with([vae])


def test_base_validation_calls_the_pure_vae_validator():
    runner = _runner()
    config = xFuserArgs(model="test-model")

    with mock.patch.object(
        base_model, "validate_vae_config"
    ) as validate:
        runner._validate_config(config)

    validate.assert_called_once_with(
        config, runner.capabilities, runner.settings
    )


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
        mock.patch.object(vae_manager.vae_tile_parallel, "mark") as mark,
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
        runner._vae_manager._apply_vae_tile_overlap(vae)

    tile_overlap_plan.assert_called_once_with(
        vae, 32, 64, sample_shape=(320, 1280)
    )
    apply.assert_called_once_with(vae, overlap_plan)


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
            runner._vae_manager._apply_vae_tile_overlap(vae)

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


def test_rectangular_tile_shape_reaches_every_staged_vae():
    first = SimpleNamespace(enable_tiling=mock.Mock(), decode=mock.Mock())
    second = SimpleNamespace(enable_tiling=mock.Mock(), decode=mock.Mock())
    runner = _runner(vae_tile_size_height=320, vae_tile_size_width=512)
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


def test_single_gpu_custom_overlap_installs_undispatched_tiled_decode():
    vae = SimpleNamespace()
    runner = _runner(vae_tile_overlap_height=32, vae_tile_overlap_width=64)
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
        mock.patch.object(
            vae_manager.vae_tiling, "latent_rows", side_effect=rows
        ),
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


def test_decode_guard_delegates_padding_detection_to_distvae():
    original = mock.Mock(side_effect=RuntimeError("decoder padding failure"))
    vae = SimpleNamespace(decode=original)
    runner = _runner(vae_tile_size_height=384, vae_tile_size_width=384)

    with mock.patch.object(
        vae_manager.vae_tiling, "is_tile_padding_error", return_value=True
    ) as is_padding:
        runner._vae_manager._install_vae_decode_guard(
            vae, tile_shape=(384, 384)
        )
        with pytest.raises(RuntimeError, match="--vae_tile_size_height"):
            vae.decode(object())

    is_padding.assert_called_once()


def test_decode_guard_reports_rectangular_window_and_flags():
    original = mock.Mock(side_effect=RuntimeError("decoder padding failure"))
    vae = SimpleNamespace(decode=original)
    runner = _runner(
        vae_tile_size_height=320,
        vae_tile_size_width=512,
    )

    with mock.patch.object(
        vae_manager.vae_tiling, "is_tile_padding_error", return_value=True
    ):
        runner._vae_manager._install_vae_decode_guard(
            vae, tile_shape=(320, 512)
        )
        with pytest.raises(RuntimeError) as error:
            vae.decode(object())

    message = str(error.value)
    assert "320x512" in message
    assert "--vae_tile_size_height" in message
    assert "--vae_tile_size_width" in message


def test_decode_oom_hint_reports_native_square_shape_with_axis_flags():
    vae = SimpleNamespace(use_tiling=True)
    runner = _runner()

    with mock.patch.object(
        vae_manager.vae_tiling, "tile_shape", return_value=(512, 512)
    ):
        hint = runner._vae_manager._vae_decode_oom_hint(vae)

    assert "512x512" in hint
    assert "--vae_tile_size_height" in hint
    assert "--vae_tile_size_width" in hint


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


def test_decode_oom_hint_reports_requested_equal_rectangular_dimensions():
    vae = SimpleNamespace(use_tiling=True)
    runner = _runner(vae_tile_size_height=384, vae_tile_size_width=384)

    with mock.patch.object(
        vae_manager.vae_tiling, "tile_shape", return_value=(384, 384)
    ):
        hint = runner._vae_manager._vae_decode_oom_hint(vae)

    assert "384x384" in hint
    assert "--vae_tile_size_height" in hint
    assert "--vae_tile_size_width" in hint


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
