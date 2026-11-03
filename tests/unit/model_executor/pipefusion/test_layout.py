from types import SimpleNamespace

import pytest

from xfuser.model_executor.pipefusion import PipeFusionPatchLayout


def test_layout_is_immutable_and_counts_reference_tokens():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(9, 14),
        token_counts=(8, 12),
        reference_token_counts=(1, 2),
        name="target+reference",
    )

    assert layout.token_ranges == ((0, 8), (8, 20))
    assert layout.image_tokens(1) == 14
    with pytest.raises(AttributeError):
        layout.split_dim = 2


def test_appended_reference_layout_tracks_generated_and_static_tokens_per_patch():
    layout = PipeFusionPatchLayout.with_appended_reference_tokens(
        generated_tokens=6,
        reference_tokens=4,
        num_patches=2,
        name="generated+reference",
    )

    assert layout.split_sizes == (5, 5)
    assert layout.token_counts == (5, 1)
    assert layout.reference_token_counts == (0, 4)


def test_layout_rejects_mismatched_patch_metadata():
    with pytest.raises(ValueError, match="equal length"):
        PipeFusionPatchLayout(
            split_dim=1,
            split_sizes=(2, 3),
            token_counts=(8,),
        )


def test_layout_install_restores_runtime_state_after_exception():
    resets = []
    state = SimpleNamespace(
        pp_patches_token_num=[2, 2],
        pp_patches_token_start_idx_local=[0, 2, 4],
        pp_patches_token_start_end_idx_global=[[0, 2], [2, 4]],
        _reset_recv_buffer=lambda: resets.append(True),
    )
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(3, 2),
        token_counts=(3, 2),
    )

    with pytest.raises(RuntimeError, match="stop"):
        with layout.install(state):
            assert state.pp_patches_token_num == [3, 2]
            raise RuntimeError("stop")

    assert state.pp_patches_token_num == [2, 2]
    assert not hasattr(state, "_xdit_pipefusion_layout_fingerprint")
    assert resets == [True, True]
