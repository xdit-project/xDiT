from typing import Callable, List, Optional, Type, TypeVar

from xfuser.core.distributed.attention_backend import AttentionBackendType, env_info

T = TypeVar("T", bound="AttentionSchedule")


class AttentionSchedule:
    """
    Per-step attention schedule defined by an explicit list of backends.
    backends[i] is the backend used at step i; len(backends) equals total_steps.
    """

    def __init__(self, backends: List[AttentionBackendType]):
        if not backends:
            raise ValueError("AttentionSchedule requires at least one step.")
        self.backends = list(backends)
        self.total_steps = len(self.backends)

    @classmethod
    def from_comma_delimited_string(cls: Type[T], s: str) -> T:
        """
        Create an AttentionSchedule from a comma-delimited string of backend names.
        Each element is interpreted as an AttentionBackendType name (case-insensitive).
        Example: "FLASH_3,FLASH_3_FP8,FLASH_3_FP8,FLASH_3"
        """
        if not s or not s.strip():
            raise ValueError("Comma-delimited string must contain at least one backend name.")
        valid_names = [e.name for e in AttentionBackendType]
        backends: List[AttentionBackendType] = []
        for token in s.split(","):
            name = token.strip().upper()
            if not name:
                raise ValueError("Empty backend name in comma-delimited string.")
            try:
                backends.append(AttentionBackendType[name])
            except KeyError:
                raise ValueError(
                    f"Unknown attention backend '{token.strip()}'. "
                    f"Valid names: {', '.join(valid_names)}."
                ) from None
        return cls(backends)

    def get_backend(self, step: int) -> AttentionBackendType:
        if step < 0 or step >= len(self.backends):
            raise IndexError(f"Step {step} out of range [0, {len(self.backends)}).")
        return self.backends[step]



def create_hybrid_attn_schedule(
    num_high_precision_steps: int,
    low_precision_backend: AttentionBackendType,
    high_precision_backend: AttentionBackendType,
    total_steps: int,
    check_compat: Optional[Callable[[AttentionBackendType], None]] = None,
) -> AttentionSchedule:
    """
    Create a hybrid attention schedule: high-precision attention in the middle, low-precision attention at start/end.
    If check_compat is provided, it is called for both backends before returning (e.g. to validate
    compatibility with the current parallel config); it may raise.
    """
    if check_compat is not None:
        check_compat(low_precision_backend)
        check_compat(high_precision_backend)

    num_low_precision_steps = total_steps - 2 * num_high_precision_steps
    if num_low_precision_steps < 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= 2 * num_high_precision_steps ({2 * num_high_precision_steps})."
        )
    backends = (
        [high_precision_backend] * num_high_precision_steps
        + [low_precision_backend] * num_low_precision_steps
        + [high_precision_backend] * num_high_precision_steps
    )
    return AttentionSchedule(backends)