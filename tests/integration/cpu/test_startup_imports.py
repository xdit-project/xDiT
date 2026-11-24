"""The launcher and the argument parser must load without the runtime.

``xdit --help`` and argument errors must not wait for diffusers, aiter (which
can compile kernels when it loads), and the model registry. Each case runs in
a fresh interpreter because an earlier test can already have loaded these
modules into the pytest process.
"""

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

RUNTIME_MODULES = ("xfuser.envs", "xfuser.model_executor", "xfuser.core.distributed", "diffusers", "aiter")

REPORT = textwrap.dedent(
    """
    import json, sys

    def report(exit_code=None):
        loaded = [name for name in {modules!r} if name in sys.modules]
        print("\\n" + json.dumps({{"exit_code": exit_code, "runtime_modules": loaded}}))
    """
).format(modules=RUNTIME_MODULES)


def _run(code: str) -> tuple[subprocess.CompletedProcess, dict]:
    result = subprocess.run(
        [sys.executable, "-c", REPORT + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result, json.loads(result.stdout.splitlines()[-1])


def test_launcher_loads_without_the_runtime():
    _, report = _run(
        """
        import xfuser.cli
        report()
        """
    )

    assert report["runtime_modules"] == []


def test_runner_parser_builds_without_the_runtime():
    _, report = _run(
        """
        from xfuser.config import FlexibleArgumentParser, xFuserArgs
        xFuserArgs.add_runner_args(FlexibleArgumentParser()).parse_args(["--model", "m"])
        report()
        """
    )

    assert report["runtime_modules"] == []


@pytest.mark.parametrize(
    ("argv", "exit_code", "message"),
    [
        (["--help"], 0, "usage:"),
        ([], 2, "the following arguments are required: --model"),
    ],
    ids=["help", "missing_model"],
)
def test_runner_exits_on_arguments_before_it_loads_the_runtime(argv, exit_code, message):
    result, report = _run(
        f"""
        import runpy
        sys.argv = ["xfuser.runner", *{argv!r}]
        try:
            runpy.run_module("xfuser.runner", run_name="__main__", alter_sys=True)
        except SystemExit as exit:
            report(exit.code)
        """
    )

    assert report == {"exit_code": exit_code, "runtime_modules": []}
    assert message in result.stdout + result.stderr
