"""The test tree stays inside the directories pytest and CI actually run."""

import importlib.util
import textwrap
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_test_layout.py"
_spec = importlib.util.spec_from_file_location("check_test_layout", _SCRIPT)
_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layout)

_REPO = Path(__file__).resolve().parents[2]
_PYTEST_INI = """\
[pytest]
testpaths =
    tests/unit
    tests/integration/cpu
    tests/integration/accelerator
"""


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def test_repository_matches_the_documented_layout():
    assert _layout.find_problems(_REPO) == []


def test_a_test_module_outside_the_collected_trees_is_rejected(tmp_path):
    _write(tmp_path, "pytest.ini", _PYTEST_INI)
    _write(tmp_path, "tests/attention/test_framework.py", "def test_always_is_satisfied():\n    assert True\n")

    problems = _layout.find_problems(tmp_path)

    assert problems == [
        "tests/attention/test_framework.py is outside the test layout "
        "(tests/unit, tests/integration/cpu, tests/integration/accelerator, tests/e2e)"
    ]


def test_vendor_marks_and_gloo_must_match_a_selector(tmp_path):
    _write(tmp_path, "pytest.ini", _PYTEST_INI)
    _write(
        tmp_path,
        "tests/unit/test_marks.py",
        """\
        import pytest

        @pytest.mark.nvidia
        @pytest.mark.rocm
        def test_both_vendors():
            pass

        @pytest.mark.gloo
        def test_gloo_in_unit():
            pass

        @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
        def test_gpu_without_accelerator():
            pass
        """,
    )

    problems = _layout.find_problems(tmp_path)

    assert problems == [
        "tests/unit/test_marks.py::test_both_vendors is marked nvidia, rocm without accelerator",
        "tests/unit/test_marks.py::test_both_vendors is marked both nvidia and rocm, "
        "so neither vendor selector runs it",
        "tests/unit/test_marks.py::test_gloo_in_unit is marked gloo outside tests/integration/cpu",
        "tests/unit/test_marks.py::test_gpu_without_accelerator skips unless a GPU is present "
        "but is not marked accelerator, so the accelerator selectors never run it",
    ]


def test_accelerator_directory_implies_the_accelerator_mark(tmp_path):
    _write(tmp_path, "pytest.ini", _PYTEST_INI)
    _write(
        tmp_path,
        "tests/integration/accelerator/test_device.py",
        """\
        import pytest

        pytestmark = [pytest.mark.multi_gpu, pytest.mark.nvidia]

        def test_runs_on_nvidia():
            pass
        """,
    )
    _write(tmp_path, "tests/e2e/test_pipeline.py", "def test_pipeline():\n    pass\n")

    assert _layout.find_problems(tmp_path) == []
