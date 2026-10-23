from dataclasses import dataclass
from contextlib import contextmanager
from copy import copy
from itertools import accumulate
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PipeFusionPatchLayout:
    """Immutable description of one PipeFusion patch partition."""

    split_dim: int
    split_sizes: Tuple[int, ...]
    token_counts: Tuple[int, ...]
    reference_token_counts: Tuple[int, ...] = ()
    name: str = "spatial"

    def __post_init__(self) -> None:
        if not self.split_sizes:
            raise ValueError("PipeFusion requires at least one patch.")
        if len(self.split_sizes) != len(self.token_counts):
            raise ValueError("PipeFusion split sizes and token counts must have equal length.")
        if self.reference_token_counts and len(self.reference_token_counts) != len(self.split_sizes):
            raise ValueError("PipeFusion reference-token counts must match the patch count.")
        if any(value <= 0 for value in self.split_sizes):
            raise ValueError("PipeFusion split sizes must be positive.")
        if any(value < 0 for value in self.token_counts):
            raise ValueError("PipeFusion token counts cannot be negative.")
        if any(value < 0 for value in self.reference_token_counts):
            raise ValueError("PipeFusion reference-token counts cannot be negative.")
        if self.reference_token_counts and any(
            split != generated + reference
            for split, generated, reference in zip(
                self.split_sizes,
                self.token_counts,
                self.reference_token_counts,
            )
        ):
            raise ValueError("Reference-token split sizes must equal generated plus reference token counts.")

    @property
    def num_patches(self) -> int:
        return len(self.split_sizes)

    @property
    def token_ranges(self) -> Tuple[Tuple[int, int], ...]:
        starts = (0, *accumulate(self.token_counts))
        return tuple(zip(starts[:-1], starts[1:]))

    @property
    def fingerprint(self) -> Tuple[Any, ...]:
        return (
            self.name,
            self.split_dim,
            self.split_sizes,
            self.token_counts,
            self.reference_token_counts,
        )

    def image_tokens(self, patch_index: int) -> int:
        reference_tokens = self.reference_token_counts[patch_index] if self.reference_token_counts else 0
        return self.token_counts[patch_index] + reference_tokens

    def split(self, tensor) -> list:
        return list(tensor.split(self.split_sizes, dim=self.split_dim))

    @contextmanager
    def install(self, state):
        """Install this layout for one run and restore the previous layout."""
        wire_token_counts = tuple(self.image_tokens(index) for index in range(self.num_patches))
        token_starts = (0, *accumulate(wire_token_counts))
        values = {
            "pp_patches_token_num": list(wire_token_counts),
            "pp_patches_token_start_idx_local": list(token_starts),
            "pp_patches_token_start_end_idx_global": [
                list(bounds) for bounds in zip(token_starts[:-1], token_starts[1:])
            ],
        }
        if self.split_dim in (2, 3):
            split_starts = (0, *accumulate(self.split_sizes))
            values.update(
                {
                    "pp_patches_height": list(self.split_sizes),
                    "pp_patches_start_idx_local": list(split_starts),
                    "pp_patches_start_end_idx_global": [
                        list(bounds)
                        for bounds in zip(
                            split_starts[:-1],
                            split_starts[1:],
                        )
                    ],
                }
            )

        previous = {name: copy(getattr(state, name)) for name in values}
        previous_fingerprint = getattr(
            state,
            "_xdit_pipefusion_layout_fingerprint",
            None,
        )
        try:
            for name, value in values.items():
                setattr(state, name, value)
            if previous_fingerprint != self.fingerprint:
                state._reset_recv_buffer()
            state._xdit_pipefusion_layout_fingerprint = self.fingerprint
            yield self
        finally:
            for name, value in previous.items():
                setattr(state, name, value)
            if previous_fingerprint != self.fingerprint:
                state._reset_recv_buffer()
            if previous_fingerprint is None:
                if hasattr(state, "_xdit_pipefusion_layout_fingerprint"):
                    delattr(state, "_xdit_pipefusion_layout_fingerprint")
            else:
                state._xdit_pipefusion_layout_fingerprint = previous_fingerprint

    @classmethod
    def with_appended_reference_tokens(
        cls,
        *,
        generated_tokens: int,
        reference_tokens: int,
        num_patches: int,
        name: str,
    ) -> "PipeFusionPatchLayout":
        """Partition a generated sequence followed by static reference tokens."""
        if generated_tokens < 1:
            raise ValueError("PipeFusion requires at least one generated token.")
        if reference_tokens < 1:
            raise ValueError("Reference-token layouts require at least one reference token.")
        if num_patches < 1:
            raise ValueError("PipeFusion requires at least one patch.")

        total_tokens = generated_tokens + reference_tokens
        base, remainder = divmod(total_tokens, num_patches)
        split_sizes = tuple(base + (patch_index < remainder) for patch_index in range(num_patches))
        if any(size == 0 for size in split_sizes):
            raise ValueError("The combined generated/reference sequence is too short for the patch count.")

        starts = (0, *accumulate(split_sizes))
        generated_counts = tuple(
            max(0, min(end, generated_tokens) - min(start, generated_tokens))
            for start, end in zip(starts[:-1], starts[1:])
        )
        reference_counts = tuple(size - generated for size, generated in zip(split_sizes, generated_counts))
        return cls(
            split_dim=1,
            split_sizes=split_sizes,
            token_counts=generated_counts,
            reference_token_counts=reference_counts,
            name=name,
        )

    @classmethod
    def from_runtime_state(
        cls,
        state,
        *,
        split_dim: int = 2,
        split_sizes: Optional[Sequence[int]] = None,
        token_counts: Optional[Sequence[int]] = None,
        reference_token_counts: Optional[Sequence[int]] = None,
        name: str = "spatial",
    ) -> "PipeFusionPatchLayout":
        return cls(
            split_dim=split_dim,
            split_sizes=tuple(split_sizes if split_sizes is not None else state.pp_patches_height),
            token_counts=tuple(token_counts if token_counts is not None else state.pp_patches_token_num),
            reference_token_counts=tuple(reference_token_counts or ()),
            name=name,
        )
