"""Tests for --solattn_beta_schedule, the per-step Sol-Attn routing threshold.

One beta for a whole run assumes every denoising step is equally approximable, and measurement on
Wan 2.2 says it is not: at a matched selected-block density of about 0.44, one attention call's
relative error against exact bf16 is 0.100 at step 0, 0.157 at step 20 and 0.172 at step 39, while
a fixed beta drifts slightly sparser across that range. These pin down the plumbing that lets a
run spend its density unevenly: the spec parses to one beta per step, the runtime counts forwards,
and the Sol backend reads the step's beta rather than the flag.
"""

import pytest
import torch

from xfuser.core.attention.backends.aiter_sol import kernel
from xfuser.core.attention.spec import AttnCall
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
    runtime.step_counter = None
    runtime.forward_step = None
    for name, value in fields.items():
        setattr(runtime, name, value)
    return runtime


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


def _tracking_runtime(**fields):
    runtime = _bare_runtime(**fields)
    runtime.track_steps()
    return runtime


@pytest.fixture
def installed(monkeypatch):
    """Install a schedule on the Sol backend the way install_beta_schedule leaves it, minus the runtime."""

    def install(schedule):
        monkeypatch.setattr(kernel, "_BETA_SCHEDULE", schedule)

    return install


def _forward_beta(runtime, **attention_kwargs) -> float:
    """One transformer forward as the Sol backend sees it: the advance, then an attention call."""
    runtime.increment_step_counter()
    call = AttnCall(step=runtime.current_step(), attention_kwargs=attention_kwargs)
    return float(kernel._beta(call, "cpu"))


def test_the_counter_walks_a_guided_schedule_one_forward_at_a_time(installed):
    """The expansion is only right if the counter still advances per forward: the two must agree,
    or a guided run would walk the schedule at half speed and never reach the last beta."""
    installed(SolAttnBetaSchedule.from_spec("1.0,0.0", 2, forwards_per_step=2))
    runtime = _tracking_runtime()

    assert [_forward_beta(runtime) for _ in range(4)] == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_a_forward_reads_its_own_step_not_the_next_one(installed):
    """The advance runs at the top of the forward, so by the time attention runs the counter has
    already moved on. Reading it there would route forward 0 at forward 1's beta, and so on."""
    installed(SolAttnBetaSchedule.from_spec("0.5,0.25,0.0", 3))
    runtime = _tracking_runtime()

    assert _forward_beta(runtime) == pytest.approx(0.5)
    assert int(runtime.current_step()) == 0
    assert int(runtime.step_counter) == 1


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


def test_the_schedule_wraps_on_its_own_length(installed):
    """Nothing else need be scheduled, and then the counter only climbs; the schedule has to wrap
    it, as a second image in one process must begin at step 0 again rather than run off the end."""
    installed(SolAttnBetaSchedule.from_spec("0.5,0.25,0.0", 3))
    runtime = _tracking_runtime()

    assert [_forward_beta(runtime) for _ in range(5)] == pytest.approx([0.5, 0.25, 0.0, 0.5, 0.25])
    assert int(runtime.step_counter) == 5


def test_a_hybrid_schedule_still_wraps_the_counter():
    runtime = _tracking_runtime(schedule_total_steps=torch.tensor(2, dtype=torch.int))

    for _ in range(3):
        runtime.increment_step_counter()

    assert int(runtime.current_step()) == 0
    assert int(runtime.step_counter) == 1


def test_a_warmup_does_not_shift_the_schedule_the_measured_run_walks(installed):
    """Each pipeline invocation starts at the first beta, whatever the invocation before it spent.

    The counter only advances, and a compile warmup runs its own forwards -- usually with a
    shortened step count. So the measured run used to start partway along the ramp with its last
    steps wrapped back onto the beginning: silent, and in the worst direction, since it hands every
    step a beta meant for a later one while the early steps are the ones whose error propagates
    through the whole run. Wan 2.2 warmed up a full cycle and landed back on zero by arithmetic,
    which is why this went unnoticed there.
    """
    schedule = SolAttnBetaSchedule.from_spec("-0.5:1.0", 8)
    installed(schedule)
    runtime = _tracking_runtime()

    for _ in range(3):  # a three-step warmup before the real thing
        _forward_beta(runtime)

    runtime.reset_step_counter()
    assert runtime.current_step() is None, "between runs there is no forward to read a step for"

    assert _forward_beta(runtime) == pytest.approx(schedule.get_beta(0))


def test_nothing_is_counted_until_something_tracks_steps():
    """Every model calls these on every forward and invocation, and most runs schedule nothing."""
    runtime = _bare_runtime()

    runtime.increment_step_counter()
    runtime.reset_step_counter()

    assert runtime.step_counter is None
    assert runtime.current_step() is None


def test_tracking_twice_keeps_the_count():
    """A beta schedule and a hybrid schedule share one counter; the second must not restart it."""
    runtime = _tracking_runtime()
    runtime.increment_step_counter()

    runtime.track_steps()

    assert int(runtime.step_counter) == 1


def test_installing_a_schedule_tracks_steps_and_clearing_it_restores_the_flag(monkeypatch):
    runtime = _bare_runtime()
    monkeypatch.setattr(kernel, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(kernel, "_BETA_SCHEDULE", None)

    kernel.install_beta_schedule("1.0:0.0", 2, forwards_per_step=2)

    assert runtime.step_counter is not None
    assert kernel._BETA_SCHEDULE.betas == [1.0, 1.0, 0.0, 0.0]

    kernel.clear_beta_schedule()

    assert _forward_beta(runtime, solattn_beta=0.125) == pytest.approx(0.125)


def test_advancing_the_step_costs_no_graph_break(installed):
    """A beta schedule alone must leave the compiled forward in one graph.

    A hybrid schedule breaks it anyway, since picking a backend or a precision is a host decision,
    but nothing in a beta-only run needs the step on the host. Subscripting the beta table with
    the 0-d step would: it lowers to select(), which needs the step as a Python int, and broke the
    graph twice per traced frame. So this compiles the real call and counts. Stands in for the transformer forward, which is where
    the advance is called from, hence where a break splits a graph that should be one.
    """
    from torch._dynamo.utils import counters

    installed(SolAttnBetaSchedule.from_spec("1.0:0.0", 4))
    runtime = _tracking_runtime()
    # The counter lives on the host and the lookup moves it to the routing's device, so a device
    # here is the case that bites: a table or a step left on the host is a host pointer in the
    # device kernel inductor fuses the lookup into -- an aborted launch, not a slow one.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(x):
        runtime.increment_step_counter()
        return x * 2 + kernel._beta(AttnCall(step=runtime.current_step()), x.device)

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(forward)
    betas = [float(compiled(torch.ones(4, device=device))[0]) - 2.0 for _ in range(8)]

    breaks = sum(counters["graph_break"].values())
    assert breaks == 0, f"{breaks} graph break(s) advancing the step: " + "; ".join(
        reason.splitlines()[0] for reason in counters["graph_break"]
    )
    # Compiled and eager have to walk the same schedule, and wrap at the same place.
    assert betas == pytest.approx(SolAttnBetaSchedule.from_spec("1.0:0.0", 4).betas * 2)


def test_the_scheduled_beta_beats_the_callers_dict(installed):
    """The dict is built once per run from --solattn_beta, so if it won, every step would share a
    beta -- the thing the schedule exists to stop."""
    installed(SolAttnBetaSchedule.from_spec("-0.25", 1))

    call = AttnCall(step=torch.tensor(0, dtype=torch.int), attention_kwargs={"solattn_beta": 0.5})

    assert float(kernel._beta(call, "cpu")) == pytest.approx(-0.25)


def test_without_a_schedule_the_flag_still_decides(installed):
    installed(None)

    call = AttnCall(step=torch.tensor(3, dtype=torch.int), attention_kwargs={"solattn_beta": 0.125})

    assert kernel._beta(call, "cpu") == pytest.approx(0.125)


def test_a_call_without_a_step_takes_the_flag(installed):
    """The attention functions are reached directly, from tests and kernel-level callers, with no
    runtime and so no step; for those "no schedule" is the answer rather than an assertion."""
    installed(SolAttnBetaSchedule.from_spec("1.0:0.0", 4))

    assert kernel._beta(AttnCall(attention_kwargs={"solattn_beta": 0.75}), "cpu") == pytest.approx(0.75)
    assert kernel._beta(AttnCall(), "cpu") == pytest.approx(0.5)
