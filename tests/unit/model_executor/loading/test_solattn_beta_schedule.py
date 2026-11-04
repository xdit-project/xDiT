"""Tests for --solattn_beta_schedule, the per-step Sol-Attn routing threshold.

One beta for a whole run assumes every denoising step is equally approximable, and measurement on
Wan 2.2 says it is not: at a matched selected-block density of about 0.44, one attention call's
relative error against exact bf16 is 0.100 at step 0, 0.157 at step 20 and 0.172 at step 39, while
a fixed beta drifts slightly sparser across that range. These pin down the plumbing that lets a
run spend its density unevenly: the spec parses to one beta per step, the step counter walks it,
and the Sol backends read the step's value rather than the flag.
"""

import pytest
import torch

from xfuser.core.distributed.attention_schedule import SolAttnBetaSchedule
from xfuser.core.distributed.runtime_state import DiTRuntimeState


def _bare_runtime(**fields):
    """A runtime state with only the fields increment_step_counter touches.

    Built without __init__ on purpose: the real one wants a pipeline and an engine config, and the
    step counter is a self-contained piece of it that should be testable without either.
    """
    runtime = object.__new__(DiTRuntimeState)
    runtime.attention_schedule = None
    runtime.schedule_total_steps = None
    runtime.gemm_schedule = None
    runtime.gemm_schedule_total_steps = None
    runtime.solattn_beta_schedule = None
    runtime.solattn_beta_schedule_total_steps = None
    runtime.scheduled_solattn_beta = None
    runtime.step_counter = None
    for name, value in fields.items():
        setattr(runtime, name, value)
    return runtime


class _SyncTrap(torch.Tensor):
    """A schedule total that refuses to be read as a Python number.

    Stands in for the real total where a test needs to prove the per-step path never reads one. A
    plain tensor cannot show that: int() on it quietly succeeds, and the cost -- a graph break in
    the compiled forward, once per forward -- is invisible from the test. Arithmetic still works,
    which is the whole of what the step path is allowed to do with it.
    """

    @staticmethod
    def __new__(cls, steps: int) -> "_SyncTrap":
        return torch.Tensor._make_subclass(cls, torch.tensor(steps, dtype=torch.int))

    def _refuse(self, how: str):
        raise AssertionError(
            f"the per-step path read a schedule total on the host via {how}, which breaks the "
            "compiled forward's graph every step"
        )

    def __int__(self):
        self._refuse("int()")

    def __index__(self):
        self._refuse("__index__()")

    def __float__(self):
        self._refuse("float()")

    def __bool__(self):
        self._refuse("bool()")

    def item(self):
        self._refuse("item()")


def test_a_ramp_hits_both_endpoints_and_moves_monotonically():
    """'first:last' is the form worth trying first, so its ends have to be exactly what was asked."""
    schedule = SolAttnBetaSchedule.from_spec("1.0:-0.25", 8)

    assert schedule.total_steps == 8
    assert schedule.get_beta(0) == pytest.approx(1.0)
    assert schedule.get_beta(7) == pytest.approx(-0.25)
    assert schedule.betas == sorted(schedule.betas, reverse=True)


def test_a_ramp_runs_in_either_direction():
    """Which end of a run deserves the density is the open question, so both must be expressible."""
    rising = SolAttnBetaSchedule.from_spec("-0.25:1.0", 4)

    assert rising.get_beta(0) == pytest.approx(-0.25)
    assert rising.get_beta(3) == pytest.approx(1.0)
    assert rising.betas == sorted(rising.betas)


def test_both_branches_of_a_guided_step_get_the_same_beta():
    """Under CFG a denoising step is two forwards and the counter advances on each, so a schedule
    read per forward would route the conditional branch at one beta and the unconditional at the
    next. Not symmetrically, either: on an ascending ramp the conditional branch takes the lower
    beta at every step of the run, so it is routed more exactly than the unconditional one
    throughout, and guidance subtracts the two rather than averaging them."""
    schedule = SolAttnBetaSchedule.from_spec("-0.5:0.0", 4, forwards_per_step=2)

    assert schedule.total_steps == 8, "eight forwards for four guided steps"
    conditional, unconditional = schedule.betas[0::2], schedule.betas[1::2]
    assert conditional == unconditional
    # The ramp still spans what was asked for, over steps rather than over forwards.
    assert schedule.per_denoising_step == pytest.approx([-0.5, -0.5 + 0.5 / 3 * 1, -0.5 + 0.5 / 3 * 2, 0.0])


def test_an_explicit_list_is_read_per_step_not_per_forward():
    """The list a user writes is one beta per denoising step; the doubling under CFG is ours to do.
    Asking for 80 entries for a 40-step guided run would be asking them to write every beta twice."""
    schedule = SolAttnBetaSchedule.from_spec("1.0,0.5,0.0", 3, forwards_per_step=2)

    assert schedule.betas == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]

    with pytest.raises(ValueError, match="lists 6 betas but the run has 3 steps"):
        SolAttnBetaSchedule.from_spec("1.0,1.0,0.5,0.5,0.0,0.0", 3, forwards_per_step=2)


def test_the_counter_walks_a_guided_schedule_one_forward_at_a_time():
    """The expansion is only right if the counter still advances per forward: the two must agree,
    or a guided run would walk the schedule at half speed and never reach the last beta."""
    schedule = SolAttnBetaSchedule.from_spec("1.0,0.0", 2, forwards_per_step=2)
    runtime = _bare_runtime(
        solattn_beta_schedule=schedule,
        solattn_beta_schedule_total_steps=torch.tensor(4, dtype=torch.int),
        step_counter=torch.tensor(0, dtype=torch.int),
    )

    seen = []
    for _ in range(4):
        runtime.increment_step_counter()
        seen.append(float(runtime.scheduled_solattn_beta))

    assert seen == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_a_single_step_run_takes_the_first_beta():
    """A one-step run has no interval to interpolate over, which must not divide by zero."""
    assert SolAttnBetaSchedule.from_spec("0.5:0.1", 1).betas == [0.5]


def test_an_explicit_list_must_cover_every_step():
    """A list that stopped short would leave the rest of the run on a beta nobody chose."""
    with pytest.raises(ValueError, match="lists 2 betas but the run has 3 steps"):
        SolAttnBetaSchedule.from_spec("0.5,0.25", 3)


def test_the_two_spec_forms_cannot_be_mixed():
    with pytest.raises(ValueError, match="mixes"):
        SolAttnBetaSchedule.from_spec("1.0:0.5,0.25", 3)


def test_a_non_numeric_beta_is_named_in_the_error():
    with pytest.raises(ValueError, match="'abc', which is not a number"):
        SolAttnBetaSchedule.from_spec("abc:1.0", 3)


def test_the_step_counter_walks_the_schedule_and_wraps():
    """The backends read one scalar, so the counter is what makes it this step's beta."""
    schedule = SolAttnBetaSchedule.from_spec("0.5,0.25,0.0", 3)
    runtime = _bare_runtime(
        solattn_beta_schedule=schedule,
        solattn_beta_schedule_total_steps=torch.tensor(3, dtype=torch.int),
        step_counter=torch.tensor(0, dtype=torch.int),
    )

    seen = []
    for _ in range(5):
        runtime.increment_step_counter()
        seen.append(runtime.scheduled_solattn_beta)

    # Five advances over a three-step run: the schedule restarts, as a second image in one process
    # has to begin at step 0 again rather than run off the end.
    assert seen == [0.5, 0.25, 0.0, 0.5, 0.25]


def test_a_beta_schedule_alone_advances_the_counter():
    """Nothing else need be scheduled. The counter used to move only for a backend or GEMM
    schedule, so a beta-only run would have sat on step 0's beta forever."""
    runtime = _bare_runtime(
        solattn_beta_schedule=SolAttnBetaSchedule.from_spec("1.0:0.0", 2),
        solattn_beta_schedule_total_steps=torch.tensor(2, dtype=torch.int),
        step_counter=torch.tensor(0, dtype=torch.int),
    )

    runtime.increment_step_counter()
    runtime.increment_step_counter()

    assert runtime.scheduled_solattn_beta == pytest.approx(0.0)


def test_a_warmup_does_not_shift_the_schedule_the_measured_run_walks():
    """Each pipeline invocation starts at the first beta, whatever the invocation before it spent.

    The counter only advances, wrapping on the schedule length, and a compile warmup runs its own
    forwards -- usually with a shortened step count. So the measured run used to start partway
    along the ramp with its last steps wrapped back onto the beginning: silent, and in the worst
    direction, since it hands every step a beta meant for a later one while the early steps are
    the ones whose error propagates through the whole run. Wan 2.2 warmed up a full cycle and
    landed back on zero by arithmetic, which is why this went unnoticed there.
    """
    schedule = SolAttnBetaSchedule.from_spec("-0.5:1.0", 8)
    runtime = _bare_runtime(
        solattn_beta_schedule=schedule,
        solattn_beta_schedule_total_steps=torch.tensor(8, dtype=torch.int),
        step_counter=torch.tensor(0, dtype=torch.int),
    )

    for _ in range(3):  # a three-step warmup before the real thing
        runtime.increment_step_counter()
    assert float(runtime.scheduled_solattn_beta) != pytest.approx(schedule.get_beta(0))

    runtime.reset_step_counter()
    runtime.increment_step_counter()

    assert float(runtime.scheduled_solattn_beta) == pytest.approx(schedule.get_beta(0))


def test_resetting_the_counter_without_a_schedule_is_harmless():
    """Every model calls this on every invocation, and most runs schedule nothing at all."""
    runtime = _bare_runtime()

    runtime.reset_step_counter()

    assert runtime.step_counter is None


def test_schedules_that_disagree_about_the_step_count_are_refused():
    """Two schedules on one counter have to be the same length or one of them is being misread.
    Caught here at setup, not per step: the per-step path runs inside the compiled forward, where
    reading these totals on the host would break the graph."""
    runtime = _bare_runtime(schedule_total_steps=torch.tensor(8, dtype=torch.int))

    with pytest.raises(RuntimeError, match="attention schedule has 8"):
        runtime.set_solattn_beta_schedule(SolAttnBetaSchedule.from_spec("1.0:0.0", 4), total_steps=4)


def test_advancing_the_step_never_reads_a_schedule_total_on_the_host():
    """The counter advance is called from the compiled transformer forward. Converting any of these
    tensors to a Python number there costs a graph break every forward, so the step path must not
    touch them beyond passing them to torch."""
    runtime = _bare_runtime(
        solattn_beta_schedule=SolAttnBetaSchedule.from_spec("1.0:0.0", 4),
        solattn_beta_schedule_total_steps=_SyncTrap(4),
        schedule_total_steps=_SyncTrap(4),
        step_counter=torch.tensor(0, dtype=torch.int),
    )

    runtime.increment_step_counter()

    assert runtime.scheduled_solattn_beta == pytest.approx(1.0)


def test_advancing_the_step_costs_no_graph_break():
    """The one that matters, and the only one that can see the whole path at once.

    _SyncTrap above catches a host read of the totals, but not every host read is a read of those:
    subscripting the beta table with the 0-d step counter lowers to select(), which needs the step
    as a Python int and broke the graph twice per traced frame while every other test here passed.
    So this compiles the real call and counts. Stands in for the transformer forward, which is
    where the advance is called from, hence where a break splits a graph that should be one.
    """
    from torch._dynamo.utils import counters

    runtime = _bare_runtime()
    # Installed the way a run installs it, rather than by setting the fields: which device the
    # counter lands on is part of what is being tested, and that is the setter's decision.
    runtime.set_solattn_beta_schedule(SolAttnBetaSchedule.from_spec("1.0:0.0", 4), total_steps=4)
    if torch.cuda.is_available():
        assert runtime.step_counter.is_cuda, (
            "a beta schedule's counter belongs on the accelerator, or the lookup inductor fuses "
            "into the routing kernel reads the schedule through a host pointer"
        )
    device = runtime.step_counter.device

    def forward(x):
        runtime.increment_step_counter()
        return x * 2 + runtime.scheduled_solattn_beta

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(forward)
    # On the accelerator when there is one, which is the case that bites: the beta is consumed by
    # the routing on the device, inductor fuses the schedule lookup into that kernel, and a table
    # or a step left on the host is a host pointer in a device kernel -- an aborted launch, not a
    # slow one. A CPU-only version of this test passes either way.
    betas = [float(compiled(torch.ones(4, device=device))[0]) - 2.0 for _ in range(8)]

    breaks = sum(counters["graph_break"].values())
    assert breaks == 0, f"{breaks} graph break(s) advancing the step: " + "; ".join(
        reason.splitlines()[0] for reason in counters["graph_break"]
    )
    # Compiled and eager have to walk the same schedule, and wrap at the same place.
    assert betas == pytest.approx(SolAttnBetaSchedule.from_spec("1.0:0.0", 4).betas * 2)


def test_setting_a_schedule_of_the_wrong_length_is_refused():
    runtime = _bare_runtime()
    schedule = SolAttnBetaSchedule.from_spec("1.0:0.0", 4)

    with pytest.raises(ValueError, match="covers 4 steps but the run has 40"):
        runtime.set_solattn_beta_schedule(schedule, total_steps=40)


def test_the_scheduled_beta_beats_the_callers_dict(monkeypatch):
    """The dict is built once per run from --solattn_beta, so if it won, every step would share a
    beta -- the thing the schedule exists to stop."""
    from types import SimpleNamespace

    from xfuser.core.attention.backends.aiter_sol import kernel
    from xfuser.core.distributed import runtime_state

    monkeypatch.setattr(runtime_state, "_RUNTIME", SimpleNamespace(scheduled_solattn_beta=-0.25), raising=False)

    assert kernel._beta({"solattn_beta": 0.5}) == pytest.approx(-0.25)


def test_without_a_schedule_the_flag_still_decides(monkeypatch):
    from types import SimpleNamespace

    from xfuser.core.attention.backends.aiter_sol import kernel
    from xfuser.core.distributed import runtime_state

    monkeypatch.setattr(runtime_state, "_RUNTIME", SimpleNamespace(scheduled_solattn_beta=None), raising=False)

    assert kernel._beta({"solattn_beta": 0.125}) == pytest.approx(0.125)


def test_beta_resolves_without_a_runtime_at_all(monkeypatch):
    """The attention functions are reached directly, from tests and kernel-level callers, and for
    those "no schedule" is the answer rather than an assertion."""
    from xfuser.core.attention.backends.aiter_sol import kernel
    from xfuser.core.distributed import runtime_state

    monkeypatch.setattr(runtime_state, "_RUNTIME", None, raising=False)

    assert kernel._beta({"solattn_beta": 0.75}) == pytest.approx(0.75)
    assert kernel._beta({}) == pytest.approx(0.5)
