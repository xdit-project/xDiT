import pytest


def test_row_split_keeps_rejecting_heights_the_ranks_cannot_share(flux_runtime_state):
    state = flux_runtime_state(sp_degree=2)

    with pytest.raises(ValueError, match="not divisible by the number of sequence parallel devices"):
        state.set_input_parameters(height=1040, width=1040, batch_size=1, num_inference_steps=4)


def test_token_sharding_callers_accept_heights_the_ranks_cannot_share_by_row(flux_runtime_state):
    state = flux_runtime_state(sp_degree=2)

    state.set_input_parameters(
        height=1040,
        width=1040,
        batch_size=1,
        num_inference_steps=4,
        split_latents_by_rows=False,
    )

    assert (state.input_config.height, state.input_config.width) == (1040, 1040)
    assert state.num_pipeline_patch == 1
    assert state.pp_patches_height is None


def test_switching_to_row_split_revalidates_an_unchanged_size(flux_runtime_state):
    state = flux_runtime_state(sp_degree=2)
    state.set_input_parameters(height=1040, width=1040, batch_size=1, split_latents_by_rows=False)

    with pytest.raises(ValueError, match="not divisible"):
        state.set_input_parameters(height=1040, width=1040, batch_size=1, split_latents_by_rows=True)


def test_row_split_metadata_follows_the_requested_height(flux_runtime_state):
    state = flux_runtime_state(sp_degree=3)

    state.set_input_parameters(height=1056, width=1056, batch_size=1, split_latents_by_rows=False)

    # 66 token rows over 3 ranks.
    assert state.pp_patches_height == [22]
    assert state.pp_patches_token_num == [22 * 66]
