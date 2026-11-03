from copy import deepcopy
from typing import Any, Iterable

import torch


class PipeFusionPatchSchedulerBank:
    """Clone a scheduler into independent patch-local state machines."""

    def __init__(
        self,
        scheduler,
        layout,
        *,
        split_state_fields: Iterable[str] = (),
    ):
        split_state_fields = tuple(split_state_fields)
        missing_fields = [field_name for field_name in split_state_fields if not hasattr(scheduler, field_name)]
        if missing_fields:
            raise TypeError(
                f"{type(scheduler).__name__} does not expose patch-splittable state: {', '.join(missing_fields)}."
            )
        self.schedulers = []
        for patch_index in range(layout.num_patches):
            patch_scheduler = deepcopy(scheduler)
            for field_name in split_state_fields:
                state = getattr(scheduler, field_name)
                setattr(
                    patch_scheduler,
                    field_name,
                    self._split_state(state, layout, patch_index),
                )
            self.schedulers.append(patch_scheduler)

    @classmethod
    def _split_state(cls, state: Any, layout, patch_index: int):
        if isinstance(state, torch.Tensor):
            return layout.split(state)[patch_index].clone()
        if isinstance(state, list):
            return [cls._split_state(value, layout, patch_index) for value in state]
        if isinstance(state, tuple):
            return tuple(cls._split_state(value, layout, patch_index) for value in state)
        if isinstance(state, dict):
            return {key: cls._split_state(value, layout, patch_index) for key, value in state.items()}
        return deepcopy(state)

    def step(self, patch_index: int, *args, **kwargs):
        return self.schedulers[patch_index].step(*args, **kwargs)
