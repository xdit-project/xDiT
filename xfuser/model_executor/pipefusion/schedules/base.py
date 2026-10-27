from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

import torch

from ..transport import PipeFusionTransport, PipeFusionWorkItem


PreparedPatch = TypeVar("PreparedPatch")
StageOutput = TypeVar("StageOutput")
Result = TypeVar("Result")


def _never_interrupted() -> bool:
    return False


def _ignore_step(step_index: int, timestep):
    return None


def pipefusion_should_update_progress(
    step_index: int,
    *,
    num_async_steps: int,
    pipeline_warmup_steps: int,
    num_warmup_steps: int,
    scheduler_order: int,
) -> bool:
    """Return whether an async PipeFusion step completes scheduler progress."""
    step = step_index + pipeline_warmup_steps + 1
    return step_index == num_async_steps - 1 or (step > num_warmup_steps and step % scheduler_order == 0)


@dataclass
class PipeFusionAsyncCallbacks(Generic[PreparedPatch, StageOutput, Result]):
    """Functional adapter for model-specific PipeFusion schedule hooks."""

    prepare_patch_fn: Callable[
        [PipeFusionWorkItem, object, Optional[torch.Tensor]],
        PreparedPatch,
    ]
    forward_patch_fn: Callable[
        [PipeFusionWorkItem, object, PreparedPatch],
        StageOutput,
    ]
    commit_patch_fn: Callable[
        [PipeFusionWorkItem, object, StageOutput],
        Optional[torch.Tensor],
    ]
    finalize_fn: Callable[[], Result]
    interrupted_fn: Callable[[], bool] = _never_interrupted
    begin_step_fn: Callable[[int, object], None] = _ignore_step
    end_step_fn: Callable[[int, object], None] = _ignore_step

    def interrupted(self) -> bool:
        return self.interrupted_fn()

    def begin_step(self, step_index: int, timestep) -> None:
        self.begin_step_fn(step_index, timestep)

    def prepare_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        received: Optional[torch.Tensor],
    ) -> PreparedPatch:
        return self.prepare_patch_fn(work, timestep, received)

    def forward_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        prepared: PreparedPatch,
    ) -> StageOutput:
        return self.forward_patch_fn(work, timestep, prepared)

    def commit_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        output: StageOutput,
    ) -> Optional[torch.Tensor]:
        return self.commit_patch_fn(work, timestep, output)

    def end_step(
        self,
        step_index: int,
        timestep,
    ) -> None:
        self.end_step_fn(step_index, timestep)

    def finalize(self) -> Result:
        return self.finalize_fn()


class PipeFusionAsyncDriver(Generic[PreparedPatch, StageOutput, Result]):
    """Model-agnostic receive/compute/send PipeFusion state machine."""

    def __init__(
        self,
        *,
        transport: PipeFusionTransport,
        hooks: PipeFusionAsyncCallbacks[PreparedPatch, StageOutput, Result],
        advance_patch,
    ):
        self.transport = transport
        self.hooks = hooks
        self.advance_patch = advance_patch

    def run(self, timesteps: Sequence) -> Result:
        num_steps = len(timesteps)
        if num_steps != self.transport.num_steps:
            raise ValueError(
                "PipeFusion transport timestep count does not match the driver: "
                f"{self.transport.num_steps} != {num_steps}."
            )
        self.transport.queue_receives()
        for step_index, timestep in enumerate(timesteps):
            if self.hooks.interrupted():
                continue
            self.hooks.begin_step(step_index, timestep)
            for patch_index in range(self.transport.num_patches):
                work = PipeFusionWorkItem(
                    step_index=step_index,
                    patch_index=patch_index,
                    num_steps=num_steps,
                    num_patches=self.transport.num_patches,
                    first_stage=self.transport.first_stage,
                )
                received = self.transport.receive(work)
                prepared = self.hooks.prepare_patch(work, timestep, received)
                output = self.hooks.forward_patch(work, timestep, prepared)
                outbound = self.hooks.commit_patch(work, timestep, output)
                final_patch = patch_index == self.transport.num_patches - 1
                if final_patch:
                    # Prime the circular feedback receive before both stages
                    # enqueue their final-patch sends. Posting both sends first
                    # can deadlock two-rank NCCL process groups.
                    self.transport.prefetch(work)
                if outbound is not None:
                    self.transport.send(outbound, patch_index)
                if not final_patch:
                    self.transport.prefetch(work)
                self.advance_patch()
            self.hooks.end_step(step_index, timestep)
        self.transport.wait_sends()
        return self.hooks.finalize()
