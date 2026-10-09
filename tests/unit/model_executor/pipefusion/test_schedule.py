import pytest
import torch

from xfuser.model_executor.pipefusion import (
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    PipeFusionImagePatchSchedule,
    PipeFusionPatchLayout,
    PipeFusionTransport,
    pipefusion_should_update_progress,
)

from tests.unit.model_executor.pipefusion._helpers import Group


def test_async_driver_computes_every_patch_without_a_step_cache():
    group = Group([torch.tensor(float(value)) for value in (10, 11, 20, 21)])
    forwarded = []
    hooks = PipeFusionAsyncCallbacks(
        prepare_patch_fn=lambda _work, _timestep, received: received,
        forward_patch_fn=lambda work, _timestep, prepared: (
            forwarded.append((work.step_index, work.patch_index)) or prepared + 1
        ),
        commit_patch_fn=lambda _work, _timestep, output: output,
        finalize_fn=lambda: "done",
    )

    result = PipeFusionAsyncDriver(
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=2,
            num_patches=2,
        ),
        hooks=hooks,
        advance_patch=lambda: None,
    ).run((100, 200))

    assert result == "done"
    assert forwarded == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [sent.item() for _, _, sent in group.sent] == [11, 12, 21, 22]
    assert all(work.waited for work in group.works)


def test_async_driver_prefetches_feedback_before_final_patch_send():
    group = Group([torch.tensor(10.0), torch.tensor(20.0)])
    hooks = PipeFusionAsyncCallbacks(
        prepare_patch_fn=lambda _work, _timestep, received: received,
        forward_patch_fn=lambda _work, _timestep, prepared: prepared + 1,
        commit_patch_fn=lambda _work, _timestep, output: output,
        finalize_fn=lambda: None,
    )

    PipeFusionAsyncDriver(
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=1,
            num_patches=2,
        ),
        hooks=hooks,
        advance_patch=lambda: None,
    ).run((1,))

    assert group.events == ["prefetch", "send-0", "prefetch", "send-1"]


def test_intermediate_stage_reuses_patch_zero_input_condition_without_resending_output():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1, 1),
        token_counts=(1, 1),
    )
    group = Group(
        [
            torch.tensor([[[10.0], [20.0]]]),
            torch.tensor([[[11.0]]]),
        ]
    )
    seen_conditions = []
    schedule = PipeFusionImagePatchSchedule(
        patch_latents=[None, None],
        layout=layout,
        initial_condition=torch.zeros(1, 1, 1),
        first_stage=False,
        last_stage=False,
        condition_reuse=True,
        forward_patch_fn=lambda _work, _time, image, condition: (
            image + 1,
            seen_conditions.append(condition.clone()) or condition + 1,
        ),
        update_last_patch_fn=lambda _work, _time, image, _previous: image,
        finalize_fn=lambda: "done",
        model_name="test",
    )

    assert (
        schedule.run(
            timesteps=(1,),
            transport=PipeFusionTransport(
                group,
                first_stage=False,
                num_steps=1,
                num_patches=2,
            ),
            advance_patch=lambda: None,
        )
        == "done"
    )
    torch.testing.assert_close(seen_conditions[0], torch.tensor([[[20.0]]]))
    torch.testing.assert_close(seen_conditions[1], torch.tensor([[[20.0]]]))
    assert [tensor.shape[1] for _, _, tensor in group.sent] == [2, 1]


def test_image_schedule_updates_generated_tokens_and_preserves_reference_tokens():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(2, 2),
        token_counts=(2, 1),
        reference_token_counts=(0, 1),
    )
    patch_latents = [
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[[3.0], [99.0]]]),
    ]
    schedule = PipeFusionImagePatchSchedule(
        patch_latents=patch_latents,
        layout=layout,
        initial_condition=torch.zeros(1, 1, 1),
        first_stage=True,
        last_stage=True,
        condition_reuse=False,
        forward_patch_fn=lambda _work, _time, image, condition: (image + 10, condition),
        update_last_patch_fn=lambda _work, _time, generated, _previous: generated + 1,
        finalize_fn=lambda: torch.cat(patch_latents, dim=1),
        model_name="reference-test",
    )

    result = schedule.run(
        timesteps=(1,),
        transport=PipeFusionTransport(
            Group([]),
            first_stage=True,
            num_steps=1,
            num_patches=2,
        ),
        advance_patch=lambda: None,
    )

    torch.testing.assert_close(result, torch.tensor([[[12.0], [13.0], [14.0], [99.0]]]))


@pytest.mark.parametrize(
    ("step_index", "expected"),
    [(0, False), (1, False), (2, True), (3, False), (4, True)],
)
def test_progress_updates_on_scheduler_boundaries_and_final_step(step_index, expected):
    assert (
        pipefusion_should_update_progress(
            step_index,
            num_async_steps=5,
            pipeline_warmup_steps=1,
            num_warmup_steps=2,
            scheduler_order=2,
        )
        is expected
    )
