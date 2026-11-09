"""Tests that the output of harness.run_process() reaches the console in every pytest capture mode.

Each test runs an inner pytest session in a separate process, so the inner session writes to a real pipe as in
CI, or to a pseudo-terminal as in a local terminal.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PROTOTYPE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROTOTYPE_DIR))
import terminal  # noqa: E402

_INNER_TEST = textwrap.dedent('''
    import sys
    sys.path.insert(0, {prototype_dir!r})
    import harness
    import pytest

    PROGRAM = """
    import os, sys
    print("PROGRAM-STDOUT")
    print("PROGRAM-STDERR", file=sys.stderr)
    print("PROGRAM-SIZE", *os.get_terminal_size())
    """

    @pytest.fixture(scope="module")
    def status(request):
        return harness.run_process(request, [sys.executable, "-c", PROGRAM], grace_period=1.0)

    def test_inner(status):
        print("TEST-PRINT")
        assert status == 0
''')

_PYTEST = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]


def _write_inner_test(directory):
    path = directory / "test_inner.py"
    path.write_text(_INNER_TEST.format(prototype_dir=str(_PROTOTYPE_DIR)))
    return path


def _assert_forwarded(output, size):
    rows, columns = size
    for line in ("PROGRAM-STDOUT", "PROGRAM-STDERR", f"PROGRAM-SIZE {columns} {rows}"):
        assert line in output
        # Live, not appended later: the line comes before the summary of the session.
        assert output.index(line) < output.index("passed")


@pytest.mark.parametrize(
    ("options", "test_print_visible"),
    [
        pytest.param([], False, id="default"),
        pytest.param(["-s"], True, id="-s"),
        # The visibility of TEST-PRINT in these modes is pytest behavior, not part of the requirement.
        pytest.param(["--capture=sys"], None, id="capture-sys"),
        pytest.param(["--capture=tee-sys"], None, id="capture-tee-sys"),
        pytest.param(["-p", "no:capture"], None, id="no-capture-plugin"),
    ],
)
def test_output_is_visible_without_terminal(tmp_path, options, test_print_visible):
    test_file = _write_inner_test(tmp_path)
    result = subprocess.run([*_PYTEST, str(test_file), *options], cwd=tmp_path, capture_output=True, timeout=120)
    output = result.stdout.decode()
    assert result.returncode == 0, output + result.stderr.decode()
    _assert_forwarded(output, terminal.DEFAULT_SIZE)  # PROGRAM-STDERR too: it goes to stdout of pytest.
    if test_print_visible is not None:
        assert ("TEST-PRINT" in output) == test_print_visible


def test_output_is_visible_in_a_terminal(tmp_path):
    test_file = _write_inner_test(tmp_path)
    chunks = []
    size = (37, 123)
    status = terminal.run_in_pty([*_PYTEST, str(test_file)], chunks.append, cwd=tmp_path, size=size, grace_period=1.0)
    output = b"".join(chunks).decode()
    assert status == 0, output
    _assert_forwarded(output, size)  # The program sees the size of the console of pytest.
    assert "TEST-PRINT" not in output
