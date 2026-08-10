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
from xfuser.model_executor.models.runner_models import base_model


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
        "vae_tile_size": None,
        "vae_tile_overlap": None,
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
    runner.settings = SimpleNamespace(model_name="test-model")
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
            "--vae-tile-size",
            "384",
            "--vae-tile-overlap",
            "0.125",
        ]
    )

    config = xFuserArgs.from_cli_args(parsed)

    assert config.use_parallel_vae is True
    assert config.vae_tile_size == 384
    assert config.vae_tile_overlap == 0.125


def test_base_model_uses_public_distvae_modules():
    assert base_model.vae_parallel is parallel
    assert base_model.vae_tiling is tiling
    assert base_model.vae_tile_parallel is tile_parallel


def test_xdit_declares_the_distvae_public_api_version_floor():
    xfuser_compat.declared_floor.cache_clear()
    assert xfuser_compat.declared_floor("distvae") == "0.0.0beta6"


def test_compat_loader_names_required_api_when_distvae_lacks_it(tmp_path):
    package = tmp_path / "distvae"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.0.0beta5"\n')
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
    assert "DistVAE>=0.0.0beta6" in error
    assert "public distvae.vae API" in error


def test_distvae_compat_loader_returns_normal_local_public_modules():
    loaded = xfuser_compat.load_distvae_vae()
    assert loaded == (parallel, tile_parallel, tiling)


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

    assert runner._decoding_vaes() == [first, second]

    runner.second_pipe.vae = first
    assert runner._decoding_vaes() == [first]


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
        mock.patch.object(base_model, "get_vae_parallel_group", return_value=coordinator),
        mock.patch.object(torch.distributed, "get_world_size", return_value=3),
        mock.patch.object(torch.distributed, "get_rank", return_value=1),
        mock.patch.object(
            base_model.vae_tiling,
            "supports_tile_parallel",
            side_effect=[True, False],
        ),
        mock.patch.object(base_model.vae_tile_parallel, "mark") as mark,
        mock.patch.object(
            base_model.vae_parallel,
            "parallelize_decoder",
            return_value="DecoderAdapter",
        ) as parallelize_decoder,
        mock.patch.object(
            base_model.vae_parallel,
            "parallelize_encoder",
            return_value="EncoderAdapter",
        ) as parallelize_encoder,
    ):
        runner._setup_parallel_vae()

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


def test_tile_size_and_overlap_are_applied_through_distvae():
    vae = SimpleNamespace()
    runner = _runner(vae_tile_size=384, vae_tile_overlap=0.125)
    size_plan = {"tile_sample_min_size": 384}
    overlap_plan = {"tile_overlap_factor": 0.125}

    with (
        mock.patch.object(base_model.vae_tiling, "tile_window", return_value=512),
        mock.patch.object(
            base_model.vae_tiling, "narrowest_useful_window", return_value=256
        ),
        mock.patch.object(
            base_model.vae_tiling,
            "snap_tile_window",
            return_value=(384, size_plan),
        ),
        mock.patch.object(
            base_model.vae_tiling, "tile_overlap", side_effect=[(0.25, 0.25), (0.125, 0.125)]
        ),
        mock.patch.object(
            base_model.vae_tiling,
            "tile_overlap_plan",
            return_value=overlap_plan,
        ),
        mock.patch.object(base_model.vae_tiling, "apply_tile_plan") as apply,
    ):
        assert runner._apply_vae_tile_size(vae) == 384
        runner._apply_vae_tile_overlap(vae)

    assert apply.call_args_list == [
        mock.call(vae, size_plan),
        mock.call(vae, overlap_plan),
    ]


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
            base_model.vae_tile_parallel, "context_of", return_value=context
        ),
        mock.patch.object(
            base_model.vae_tile_parallel,
            "sharing",
            return_value=("dispatch", "assemble"),
        ) as sharing,
        mock.patch.object(
            base_model.vae_tiling,
            "tiled_decode_for",
            return_value=installed,
        ) as tiled_decode_for,
    ):
        runner._install_vae_tiled_decode(vae)

    sharing.assert_called_once_with(context)
    tiled_decode_for.assert_called_once_with(vae, "dispatch", "assemble")
    assert vae.tiled_decode is installed


def test_decode_guard_delegates_padding_detection_to_distvae():
    original = mock.Mock(side_effect=RuntimeError("decoder padding failure"))
    vae = SimpleNamespace(decode=original)
    runner = _runner()

    with mock.patch.object(
        base_model.vae_tiling, "is_tile_padding_error", return_value=True
    ) as is_padding:
        runner._install_vae_decode_guard(vae, tile_window=384)
        with pytest.raises(RuntimeError, match="--vae_tile_size"):
            vae.decode(object())

    is_padding.assert_called_once()


def test_decode_oom_hint_uses_distvae_size_api():
    vae = SimpleNamespace(use_tiling=True)
    runner = _runner()

    with (
        mock.patch.object(base_model.vae_tiling, "tile_window", return_value=512),
        mock.patch.object(
            base_model.vae_tiling, "snap_tile_window", return_value=(256, {})
        ) as snap,
    ):
        hint = runner._vae_decode_oom_hint(vae)

    snap.assert_called_once_with(vae, 256)
    assert "--vae_tile_size 256" in hint


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
