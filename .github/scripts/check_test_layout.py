#!/usr/bin/env python3
"""Reject test modules the documented pytest selectors would never run.

The collected trees are ``tests/unit``, ``tests/integration/cpu``, and
``tests/integration/accelerator``. ``tests/e2e`` is the only other place a
test module may live, and it stays out of ``pytest.ini`` ``testpaths``.
"""

from __future__ import annotations

import ast
import configparser
import fnmatch
import os
import sys
from pathlib import Path

COLLECTED = (
    "tests/unit",
    "tests/integration/cpu",
    "tests/integration/accelerator",
)
ALLOWED = COLLECTED + ("tests/e2e",)
IGNORED_MARKS = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings"}
VENDOR_MARKS = {"nvidia", "rocm", "multi_gpu"}


def find_problems(root: Path) -> list[str]:
    problems = _testpaths_problems(root)
    tests_root = root / "tests"
    if not tests_root.is_dir():
        return problems + ["tests/ is missing"]

    for dirpath, dirnames, filenames in os.walk(tests_root):
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            problems.extend(_file_problems(root, path))
    return problems


def _testpaths_problems(root: Path) -> list[str]:
    parser = configparser.ConfigParser()
    read = parser.read(root / "pytest.ini")
    if not read or "pytest" not in parser or "testpaths" not in parser["pytest"]:
        return ["pytest.ini must set [pytest] testpaths"]
    found = tuple(part for part in parser["pytest"]["testpaths"].split() if part)
    if tuple(sorted(found)) != tuple(sorted(COLLECTED)):
        expected = ", ".join(COLLECTED)
        actual = ", ".join(found) or "(empty)"
        return [f"pytest.ini testpaths must be {expected}; found {actual}"]
    return []


def _file_problems(root: Path, path: Path) -> list[str]:
    relative = path.relative_to(root).as_posix()
    bucket = _bucket(relative)
    collectable = _is_test_module(path.name)
    try:
        tree = ast.parse(path.read_text(), filename=relative)
    except SyntaxError as exc:
        return [f"{relative} cannot be parsed: {exc.msg}"]

    items = _test_items(tree)
    if not collectable:
        if items and path.name != "conftest.py":
            return [f"{relative} defines tests, but pytest only collects test_*.py and *_test.py"]
        return []
    if bucket is None:
        allowed = ", ".join(ALLOWED)
        return [f"{relative} is outside the test layout ({allowed})"]
    if not items:
        return [f"{relative} is a test module with no tests"]
    if bucket == "tests/e2e":
        return []

    problems = []
    implied_accelerator = bucket == "tests/integration/accelerator"
    for name, marks, needs_gpu in items:
        node = f"{relative}::{name}"
        accelerator = implied_accelerator or "accelerator" in marks
        if "gloo" in marks and bucket != "tests/integration/cpu":
            problems.append(f"{node} is marked gloo outside tests/integration/cpu")
        if marks & VENDOR_MARKS and not accelerator:
            joined = ", ".join(sorted(marks & VENDOR_MARKS))
            problems.append(f"{node} is marked {joined} without accelerator")
        if needs_gpu and not accelerator:
            problems.append(
                f"{node} skips unless a GPU is present but is not marked accelerator, "
                "so the accelerator selectors never run it"
            )
        if "nvidia" in marks and "rocm" in marks:
            problems.append(f"{node} is marked both nvidia and rocm, so neither vendor selector runs it")
        elif not _selected(bucket, marks):
            problems.append(f"{node} matches no documented pytest selector")
    return problems


def _bucket(relative: str) -> str | None:
    for root in ALLOWED:
        if relative == root or relative.startswith(root + "/"):
            return root
    return None


def _is_test_module(filename: str) -> bool:
    return fnmatch.fnmatch(filename, "test_*.py") or fnmatch.fnmatch(filename, "*_test.py")


def _selected(bucket: str, marks: set[str]) -> bool:
    accelerator = bucket == "tests/integration/accelerator" or "accelerator" in marks
    if "nvidia" in marks and "rocm" in marks:
        return False
    if not accelerator:
        return bucket in {"tests/unit", "tests/integration/cpu"}
    return True


def _test_items(tree: ast.AST) -> list[tuple[str, set[str], bool]]:
    module_marks, module_needs_gpu = _scope_marks(tree.body)
    items = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            marks, needs_gpu = _decorator_marks(node.decorator_list)
            items.append((node.name, module_marks | marks, module_needs_gpu or needs_gpu))
        elif isinstance(node, ast.ClassDef) and _is_test_class(node):
            class_marks, class_needs_gpu = _scope_marks(node.body)
            class_decorator_marks, class_decorator_gpu = _decorator_marks(node.decorator_list)
            inherited = module_marks | class_marks | class_decorator_marks
            inherited_gpu = module_needs_gpu or class_needs_gpu or class_decorator_gpu
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name.startswith("test"):
                    marks, needs_gpu = _decorator_marks(child.decorator_list)
                    items.append((f"{node.name}.{child.name}", inherited | marks, inherited_gpu or needs_gpu))
    return items


def _is_test_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("Test"):
        return True
    return any(_name(base).endswith("TestCase") for base in node.bases)


def _scope_marks(body: list[ast.stmt]) -> tuple[set[str], bool]:
    marks: set[str] = set()
    needs_gpu = False
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        marks, needs_gpu = _mark_expr(node.value)
    return marks, needs_gpu


def _decorator_marks(decorators: list[ast.expr]) -> tuple[set[str], bool]:
    marks: set[str] = set()
    needs_gpu = False
    for decorator in decorators:
        found, gpu = _mark_expr(decorator)
        marks |= found
        needs_gpu = needs_gpu or gpu
    return marks, needs_gpu


def _mark_expr(node: ast.AST) -> tuple[set[str], bool]:
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        marks: set[str] = set()
        needs_gpu = False
        for element in node.elts:
            found, gpu = _mark_expr(element)
            marks |= found
            needs_gpu = needs_gpu or gpu
        return marks, needs_gpu
    return _one_mark(node)


def _one_mark(node: ast.AST) -> tuple[set[str], bool]:
    needs_gpu = _skips_without_cuda(node)
    call = node
    if isinstance(node, ast.Call):
        call = node.func
    if isinstance(call, ast.Attribute) and call.attr not in IGNORED_MARKS:
        parent = call.value
        if isinstance(parent, ast.Attribute) and parent.attr == "mark":
            return {call.attr}, needs_gpu
    return set(), needs_gpu


def _skips_without_cuda(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "skipif":
        return False
    skipped = node.args[0]
    if not isinstance(skipped, ast.UnaryOp) or not isinstance(skipped.op, ast.Not):
        return False
    available = skipped.operand
    if not isinstance(available, ast.Call):
        return False
    target = available.func
    if not isinstance(target, ast.Attribute) or target.attr != "is_available":
        return False
    cuda = target.value
    return isinstance(cuda, ast.Attribute) and cuda.attr == "cuda"


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[2]
    problems = find_problems(root)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
