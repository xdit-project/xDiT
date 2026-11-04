from typing import Callable, Dict, List, Optional, Type, TypeVar

import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()

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
                    f"Unknown attention backend '{token.strip()}'. Valid names: {', '.join(valid_names)}."
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


class SolAttnBetaSchedule:
    """
    Per-step Sol-Attn routing threshold, defined by an explicit list of betas.
    betas[i] is the beta used at step i; len(betas) equals total_steps.

    Beta trades exactness for speed within a single attention call: tau = mean_j(proxy) + beta *
    std_j(proxy), so a lower beta admits more KV blocks into the exact pass. A single value spends
    the same everywhere, which is only right if every denoising step is equally approximable.
    Measured on Wan 2.2 at 1104x832, it is not: at a matched selected-block density of about 0.44,
    the relative error of one attention call against exact bf16 is 0.100 at step 0, 0.157 at step
    20 and 0.172 at step 39, while a fixed beta drifts slightly SPARSER over the same range. The
    steps that approximate worst are being given the least help.

    Which end of the run deserves the density is not settled by that, though, and cannot be: an
    early error is carried through every step that follows it, a late one is not, and per-call
    error cannot weigh those against each other. Hence a schedule rather than a fixed direction.
    """

    def __init__(self, betas: List[float], forwards_per_step: int = 1):
        if not betas:
            raise ValueError("SolAttnBetaSchedule requires at least one step.")
        self.betas = [float(beta) for beta in betas]
        # Counted in forwards, because that is what the step counter counts: it advances once per
        # transformer forward, and a denoising step is forwards_per_step of them.
        self.total_steps = len(self.betas)
        self.forwards_per_step = forwards_per_step
        self._betas_tensor = torch.tensor(self.betas, dtype=torch.float32)
        # One copy per device that asks, because the schedule cannot know at construction which
        # device will walk it, and the table is total_steps floats.
        self._device_tables: Dict[torch.device, torch.Tensor] = {}

    @property
    def per_denoising_step(self) -> List[float]:
        """One beta per denoising step, dropping the repeats -- the schedule as it was asked for.

        For logging and reporting: the expanded list is the right thing to index per forward and
        the wrong thing to read, since under CFG it prints every beta twice.
        """
        return self.betas[:: self.forwards_per_step]

    def betas_on(self, device) -> torch.Tensor:
        """The whole schedule as a tensor on `device`, built once per device."""
        device = torch.device(device)
        if device not in self._device_tables:
            self._device_tables[device] = self._betas_tensor.to(device)
        return self._device_tables[device]

    @classmethod
    def from_spec(
        cls: Type["SolAttnBetaSchedule"], spec: str, total_steps: int, forwards_per_step: int = 1
    ) -> "SolAttnBetaSchedule":
        """
        Parse a schedule spec into one beta per DENOISING STEP, held across that step's forwards.

        "first:last" ramps linearly from first at step 0 to last at the final step, which is the
        shape worth trying first and unreasonable to write out by hand for a 40-step run.
        "b0,b1,..." lists every step explicitly, for any shape that is not a ramp; its length must
        be total_steps, because a list that silently stopped short would leave the rest of the run
        on a beta nobody chose.

        forwards_per_step is how many transformer forwards one denoising step costs -- 2 under
        classifier-free guidance, one for the conditional branch and one for the unconditional.
        The step counter advances on each forward, so each beta is repeated that many times, which
        is what makes a step's two branches share one. Interpolating over forwards instead would
        put the two branches one increment apart, and not symmetrically: on an ascending ramp the
        conditional branch would take the lower beta at every step of the run, so it would be
        routed more exactly than the unconditional one throughout, and guidance subtracts them --
        it does not average the bias away. Measured on a real Wan 2.2 step, one increment of a
        -0.5:0.0 ramp moves selected-block density 0.44% and flips 0.29% of routing decisions,
        against the 39.6% that differ between the branches anyway because their operands differ.
        So this is a correctness fix with a small effect, not a quality fix.

        The hybrid attention and GEMM schedules arrive at the same place from the other direction:
        they multiply their high-precision step COUNT by the same factor, which makes them
        piecewise constant in blocks of forwards_per_step and their branches agree by construction.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if forwards_per_step <= 0:
            raise ValueError(f"forwards_per_step must be positive, got {forwards_per_step}.")
        text = (spec or "").strip()
        if not text:
            raise ValueError("Sol-Attn beta schedule spec must not be empty.")

        def expanded(per_step: List[float]) -> "SolAttnBetaSchedule":
            """One beta per forward, each step's repeated across the forwards it spends."""
            return cls(
                [beta for beta in per_step for _ in range(forwards_per_step)], forwards_per_step=forwards_per_step
            )

        if ":" in text:
            if "," in text:
                raise ValueError(
                    f"Sol-Attn beta schedule {spec!r} mixes the 'first:last' ramp form with the "
                    "comma-separated per-step form; use one or the other."
                )
            parts = text.split(":")
            if len(parts) != 2:
                raise ValueError(f"Sol-Attn beta schedule ramp {spec!r} must be 'first:last'.")
            first, last = (cls._parse_beta(part, spec) for part in parts)
            if total_steps == 1:
                return expanded([first])
            step = (last - first) / (total_steps - 1)
            return expanded([first + step * index for index in range(total_steps)])

        betas = [cls._parse_beta(token, spec) for token in text.split(",")]
        if len(betas) != total_steps:
            forwards = (
                ""
                if forwards_per_step == 1
                else f" (the run spends {total_steps * forwards_per_step} forwards on them, "
                f"{forwards_per_step} per step, but the list is per step)"
            )
            raise ValueError(
                f"Sol-Attn beta schedule {spec!r} lists {len(betas)} betas but the run has "
                f"{total_steps} steps{forwards}. Give one per step, or use the 'first:last' ramp "
                "form."
            )
        return expanded(betas)

    @staticmethod
    def _parse_beta(token: str, spec: str) -> float:
        try:
            return float(token.strip())
        except ValueError:
            raise ValueError(
                f"Sol-Attn beta schedule {spec!r} contains {token.strip()!r}, which is not a number."
            ) from None

    def get_beta(self, step: int) -> float:
        """This step's beta as a number, for setup and for reporting. Not for the per-step path
        inside the compiled forward: see get_beta_tensor."""
        if step < 0 or step >= len(self.betas):
            raise IndexError(f"Step {step} out of range [0, {len(self.betas)}).")
        return self.betas[step]

    def get_beta_tensor(self, step) -> torch.Tensor:
        """This step's beta as a 0-d tensor, indexed by a step that may itself be a tensor.

        The per-step read runs inside the compiled transformer forward, and this keeps both the step
        and the beta inside the graph. Returning a Python float there costs twice over: reading the
        step counter on the host breaks the graph once per forward, and the float is then baked in
        as a constant, so the forward recompiles for every distinct beta the schedule holds -- a
        40-step ramp past the recompile limit, after which dynamo gives up and runs it eagerly.

        For the same reason the range cannot be checked here. The counter wraps on total_steps
        before this ever sees it, so an out-of-range step is not reachable.

        index_select rather than self._betas_tensor[step], which is the same lookup but not the
        same cost: subscripting with a 0-d tensor lowers to select(), whose index is a Python int,
        so it reads the step on the host and breaks the graph -- measured at 2 breaks per traced
        frame against 0 for this. as_tensor leaves a tensor step alone and lifts the plain int that
        setup passes.

        Read from the table on the STEP's device, which is what keeps the lookup single-device.
        inductor fuses it into the routing kernel that consumes the beta -- the fused kernel's name
        carries both this index_select and the threshold's std -- and a Triton kernel cannot take a
        host pointer, so a host-resident table with a device step aborts the launch with "Pointer
        argument cannot be accessed from Triton (cpu tensor?)". Hence the runtime puts the counter
        on the device where the routing runs and the table follows it here.
        """
        index = torch.as_tensor(step).reshape(1)
        return self.betas_on(index.device).index_select(0, index).squeeze(0)


class GemmPrecisionSchedule:
    """
    Per-step GEMM precision schedule defined by an explicit list of booleans.
    use_high_precision[i] indicates whether step i should use high precision GEMM.
    """

    def __init__(self, use_high_precision_schedule: List[bool]):
        if not use_high_precision_schedule:
            raise ValueError("GemmPrecisionSchedule requires at least one step.")
        self.use_high_precision_schedule = list(use_high_precision_schedule)
        self.total_steps = len(self.use_high_precision_schedule)

    @classmethod
    def from_comma_delimited_string(
        cls,
        value: str,
        *,
        low_format: str = "fp4",
        high_format: str = "fp8",
    ) -> "GemmPrecisionSchedule":
        schedule = []
        for token in value.split(","):
            name = token.strip().lower()
            if name not in {low_format, high_format}:
                raise ValueError(f"Unknown GEMM schedule format {name!r}; expected {high_format} or {low_format}.")
            schedule.append(name == high_format)
        return cls(schedule)

    def is_high_precision(self, step: int) -> bool:
        if step < 0 or step >= len(self.use_high_precision_schedule):
            raise IndexError(f"Step {step} out of range [0, {len(self.use_high_precision_schedule)}).")
        return self.use_high_precision_schedule[step]


def create_hybrid_gemm_schedule(
    num_high_precision_steps: int,
    total_steps: int,
) -> GemmPrecisionSchedule:
    """
    Create a hybrid GEMM schedule: high-precision GEMM at start/end, low-precision GEMM in the middle.
    """
    num_low_precision_steps = total_steps - 2 * num_high_precision_steps
    if num_low_precision_steps < 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= 2 * num_high_precision_steps ({2 * num_high_precision_steps})."
        )
    schedule = [True] * num_high_precision_steps + [False] * num_low_precision_steps + [True] * num_high_precision_steps
    return GemmPrecisionSchedule(schedule)
