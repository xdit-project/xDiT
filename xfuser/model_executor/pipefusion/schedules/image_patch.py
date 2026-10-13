from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, Sequence

import torch

from ..payload import CombinedTensorPayloadCodec, PipeFusionStagePayload
from ..transport import PipeFusionTransport, PipeFusionWorkItem
from .base import (
    Result,
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    _ignore_step,
    _never_interrupted,
)


@dataclass(frozen=True)
class PipeFusionPatchForward:
    """Bind shared and patch-indexed arguments to a pipeline backbone."""

    forward_fn: Callable[..., Any]
    static_kwargs: Mapping[str, Any] = field(default_factory=dict)
    patch_kwargs: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def __call__(self, work, timestep, image_state, condition_state):
        kwargs = dict(self.static_kwargs)
        kwargs.update({name: values[work.patch_index] for name, values in self.patch_kwargs.items()})
        return self.forward_fn(
            latents=image_state,
            encoder_hidden_states=condition_state,
            t=timestep,
            **kwargs,
        )


class PipeFusionImagePatchSchedule(Generic[Result]):
    """Schedule image patches with an optional propagated condition state."""

    def __init__(
        self,
        *,
        patch_latents: list[torch.Tensor],
        layout,
        initial_condition: torch.Tensor | Callable[[], torch.Tensor],
        first_stage: bool,
        last_stage: bool,
        condition_reuse: bool,
        forward_patch_fn: Callable[
            [PipeFusionWorkItem, object, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor | None],
        ],
        update_last_patch_fn: Callable[
            [PipeFusionWorkItem, object, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        finalize_fn: Callable[[], Result],
        begin_step_fn: Callable[[int, object], None] = _ignore_step,
        end_step_fn: Callable[[int, object], None] = _ignore_step,
        interrupted_fn: Callable[[], bool] = _never_interrupted,
        model_name: str = "PipeFusion",
    ):
        self.patch_latents = patch_latents
        self.layout = layout
        self.initial_condition = initial_condition
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.condition_reuse = condition_reuse
        self.forward_patch_fn = forward_patch_fn
        self.update_last_patch_fn = update_last_patch_fn
        self.finalize_fn = finalize_fn
        self.begin_step_fn = begin_step_fn
        self.end_step_fn = end_step_fn
        self.interrupted_fn = interrupted_fn
        self.model_name = model_name

    def run(
        self,
        *,
        timesteps: Sequence,
        transport: PipeFusionTransport,
        advance_patch: Callable[[], None],
    ) -> Result:
        if len(self.patch_latents) != self.layout.num_patches:
            raise ValueError(f"{self.model_name} patch latent count does not match layout.")

        payload_codec = CombinedTensorPayloadCodec(
            self.layout,
            model_name=self.model_name,
        )
        previous = [None] * self.layout.num_patches if self.last_stage else None
        condition_states = [None] * self.layout.num_patches
        incoming_condition = [None]
        outgoing_condition = [None]

        def begin_step(step_index, timestep):
            incoming_condition[0] = None
            outgoing_condition[0] = None
            self.begin_step_fn(step_index, timestep)

        def prepare_patch(work, _timestep, received):
            patch_index = work.patch_index
            if self.last_stage:
                previous[patch_index] = self.patch_latents[patch_index]
            if received is not None:
                if self.first_stage:
                    self.patch_latents[patch_index] = received
                else:
                    includes_condition = not (self.condition_reuse and patch_index > 0)
                    payload = payload_codec.unpack(
                        received,
                        patch_index,
                        includes_condition=includes_condition,
                    )
                    self.patch_latents[patch_index] = payload.image_state
                    if includes_condition:
                        condition_states[patch_index] = payload.condition_state
            if self.first_stage:
                condition = self.initial_condition() if callable(self.initial_condition) else self.initial_condition
            elif self.condition_reuse and patch_index > 0:
                condition = incoming_condition[0]
            else:
                condition = condition_states[patch_index]
            if condition is None:
                raise RuntimeError(f"{self.model_name} missing condition state for patch {patch_index}.")
            if self.condition_reuse and patch_index == 0:
                incoming_condition[0] = condition
            return PipeFusionStagePayload(
                image_state=self.patch_latents[patch_index],
                condition_state=condition,
            )

        def forward_patch(work, timestep, prepared):
            return self.forward_patch_fn(
                work,
                timestep,
                prepared.image_state,
                prepared.condition_state,
            )

        def commit_patch(work, timestep, output):
            patch_index = work.patch_index
            image_state, next_condition = output
            self.patch_latents[patch_index] = image_state
            if self.last_stage:
                generated_tokens = self.layout.token_counts[patch_index]
                generated_slice = [slice(None)] * image_state.ndim
                generated_slice[self.layout.split_dim] = slice(0, generated_tokens)
                reference_slice = [slice(None)] * image_state.ndim
                reference_slice[self.layout.split_dim] = slice(generated_tokens, None)
                updated_generated = self.update_last_patch_fn(
                    work,
                    timestep,
                    image_state[tuple(generated_slice)],
                    previous[patch_index][tuple(generated_slice)],
                )
                updated = torch.cat(
                    (
                        updated_generated,
                        previous[patch_index][tuple(reference_slice)],
                    ),
                    dim=self.layout.split_dim,
                )
                self.patch_latents[patch_index] = updated
                return None if work.step_index == work.num_steps - 1 else updated

            if self.condition_reuse:
                if patch_index == 0:
                    outgoing_condition[0] = next_condition
                next_condition = outgoing_condition[0]
            if next_condition is None:
                raise RuntimeError(f"{self.model_name} produced no condition state for patch {patch_index}.")
            return payload_codec.pack(
                PipeFusionStagePayload(
                    image_state=image_state,
                    condition_state=next_condition,
                ),
                patch_index,
                include_condition=not (self.condition_reuse and patch_index > 0),
            )

        callbacks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=self.finalize_fn,
            interrupted_fn=self.interrupted_fn,
            begin_step_fn=begin_step,
            end_step_fn=self.end_step_fn,
        )
        return PipeFusionAsyncDriver(
            transport=transport,
            hooks=callbacks,
            advance_patch=advance_patch,
        ).run(timesteps)
