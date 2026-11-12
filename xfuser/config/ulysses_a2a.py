from __future__ import annotations

from dataclasses import dataclass

# default: the process group's all-to-all (NCCL / RCCL); sdma: the copy engines; packed: the
# SDMA path's layout and head groups over the default all-to-all (internal: tests and
# diagnostics, to tell the transport's share of a gain from the layout's).
ULYSSES_A2A_MODES = ("default", "sdma", "packed")


@dataclass(frozen=True)
class UlyssesA2AConfig:
    """The Ulysses all-to-all (--ulysses_a2a) and its SDMA options (--ulysses_sdma_*), read by
    xfuser/model_executor/layers/usp_sdma.py from the runtime config."""

    mode: str = "default"
    head_groups: int = 2
    group_streams: bool = False
    overlap_qkv: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ULYSSES_A2A_MODES:
            raise ValueError(f"Ulysses all-to-all mode must be one of {ULYSSES_A2A_MODES}; got {self.mode!r}")
        if self.head_groups < 1:
            raise ValueError(f"head_groups must be at least 1; got {self.head_groups}")
