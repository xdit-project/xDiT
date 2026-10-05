'''This module is meant to be an independent from the rest of xFuser self-contained collection
of utilities for handling determinism check results, primarily for use by external
higher-level runners. It's meant to couple with the xFuser only by externally visible
artifacts it produces, such as logs or files. However, due to this coupling it has to be
a part of the xFuser, so it's maintained and doesn't go obsolete.

Due to the module being deep inside xFuser internal packages, importing it directly the
standard way like
import xfuser.core.utils.determinism_check_results
would trigger importing of all parent modules of xFuser, that typically import pytorch and
possibly other heavy dependencies. This could take a long time and produce lot's of spam to
the console. To avoid this, you could import the module manually like this:

```
def import_xfuser_determinism_check_results():
    """Imports the determinism check results functions from xfuser.
    That roundtrip is necessary instead of a simple

    ```
    from xfuser.core.utils.determinism_check_results import (
        determinism_check_results,
        readable_bytes,
    )
    ```

    to avoid importing all parent modules of xfuser, that typically import pytorch,
    which takes a long time and could produce lot's of spam to the console.
    """
    import importlib.util
    from pathlib import Path

    package = importlib.util.find_spec("xfuser")
    assert package is not None, "xfuser package not found"
    package_dir = Path(next(iter(package.submodule_search_locations)))
    path = package_dir / "core/utils/determinism_check_results.py"

    spec = importlib.util.spec_from_file_location(
        "_xfuser_core_utils_determinism_check_results", path
    )
    assert spec is not None, "xfuser.core.utils.determinism_check_results module spec not found"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.determinism_check_results, module.readable_bytes

determinism_check_results, readable_bytes = import_xfuser_determinism_check_results()
```
'''

import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

_FAILED_LOG_PATTERN = re.compile(
    r"determinism_check\[rank [0-9]+\]: iteration [0-9]+ diverged"
)
_FAILED_OUTPUT_PATTERN = re.compile(r"_rank_([0-9]+)_iteration_([0-9]+)")


def _count_determinism_failures_in_log(log_file: str | Path) -> int:
    failure_count = 0
    with open(log_file, encoding="utf-8") as stream:
        for line in stream:
            failure_count += len(_FAILED_LOG_PATTERN.findall(line))
    return failure_count


def _failed_output_statistics(directory: str | Path) -> tuple[int, int] | None:
    failed_output_events: set[tuple[int, int]] = set()
    failed_output_size = 0
    found_failed_output = False

    with os.scandir(directory) as scanner:
        for entry in scanner:
            if not entry.is_file(follow_symlinks=False):
                continue

            output_match = _FAILED_OUTPUT_PATTERN.search(entry.name)
            if output_match is None:
                continue

            found_failed_output = True
            failed_output_size += entry.stat(follow_symlinks=False).st_size
            rank = int(output_match.group(1))
            iteration = int(output_match.group(2))
            if iteration != 0:
                failed_output_events.add((rank, iteration))

    if not found_failed_output:
        return None
    return len(failed_output_events), failed_output_size


def determinism_check_results(
    dir_tree: str | Path,
    expected_log_files: str | Iterable[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """For a given directory tree containing output directories as leaves, return a collection
    of output directories (as keys of the returned dictionary having paths relative to the tree
    root) that have determinism check failed, along with the number of failed checks and the total
    size in bytes of dumped files related to the failed checks (as the dictionary's value, a tuple
    of two integers in that order).

    Args:
        dir_tree: The root directory of the directory tree to check.
        expected_log_files: name of log files to expect in the output directories. Empty string or
            iterable disables checking in log files, None defaults to ["stdout.txt", "stderr.txt"].

    Algorithm:
    - Traverse the directory tree recursively, following symlinks (preventing infinite loops), and
    assume each internal directory that doesn't have subdirectories to be a model config's output
    directory.
    - For each such output directory found perform two actions:
        1. If it contains any of the `expected_log_files`, scan each of them to count occurrences of
            a pattern corresponding to a failed determinism check error message:
                `determinism_check[rank {rank}]: iteration {iteration + 1} diverged`
            and then sum them up across all log files (otherwise assume 0).
        2. Scan the directory for presence of files the pattern of names generated by
            `xFuserModel._save_determinism_check_failed_outputs()`:
                `_rank_{rank:d}_iteration_{iteration:d}`
            and sum up their total size in bytes, and count the unique `(rank, iteration)` pairs
            among such files, excluding pairs corresponding to the 0-th iteration. Otherwise
            assume 0 and 0.
        Whenever any of the above actions finds at least one failure, add relative to the root path
        to the directory to the result dictionary as a key, and the tuple of
        `(max(number_of_failed_checks_from_logs, number_of_failed_checks_from_files),
        total_size_of_failed_output_files_in_bytes)` as the value.
    """
    if not isinstance(dir_tree, Path):
        dir_tree = Path(dir_tree)
    if expected_log_files is None:
        expected_log_files = ["stdout.txt", "stderr.txt"]
    if isinstance(expected_log_files, str):
        if expected_log_files:
            expected_log_files = frozenset((expected_log_files,))
        else:
            expected_log_files = frozenset()
    else:
        expected_log_files = frozenset(expected_log_files)

    for log_file_name in expected_log_files:
        if not isinstance(log_file_name, str):
            raise TypeError("expected_log_files entries must be strings")
        log_file_path = Path(log_file_name)
        if (
            not log_file_name
            or log_file_path.is_absolute()
            or len(log_file_path.parts) != 1
            or log_file_path.name != log_file_name
            or log_file_name in (".", "..")
        ):
            raise ValueError(
                "expected_log_files entries must be direct-child filenames"
            )

    root_stat = dir_tree.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"Not a directory: {dir_tree}")

    results: dict[str, tuple[int, int]] = {}
    stack: list[tuple[Path, frozenset[tuple[int, int]]]] = [(dir_tree, frozenset())]

    while stack:
        directory, ancestors = stack.pop()
        directory_stat = directory.stat()
        directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
        if directory_identity in ancestors:
            continue

        with os.scandir(directory) as scanner:
            entries = list(scanner)
        subdirectories = [
            entry for entry in entries if entry.is_dir(follow_symlinks=True)
        ]

        if subdirectories:
            child_ancestors = ancestors | {directory_identity}
            stack.extend(
                (directory / entry.name, child_ancestors)
                for entry in reversed(subdirectories)
            )
            continue

        failed_logs = 0

        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue

            if entry.name in expected_log_files:
                failed_logs += _count_determinism_failures_in_log(entry.path)

        failed_stats = _failed_output_statistics(directory)
        if failed_logs or failed_stats is not None:
            failed_events, failed_size = failed_stats or (0, 0)
            rel_dir = directory.relative_to(dir_tree)
            result_key = "" if rel_dir == Path(".") else str(rel_dir)
            results[result_key] = (max(failed_logs, failed_events), failed_size)

    return results


def readable_bytes(size: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024 / 1024 / 1024:.1f} GB"
