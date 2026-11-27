"""Parse /ci-rocm comments without evaluating their contents."""

import argparse
import json
import os
from pathlib import Path
import shlex

DEFAULT_REQUEST = {
    "benchmarks": "--tag nightly",
    "architectures": "gfx942,gfx950",
    "amd_ref": "main",
    "downstream_configs": "false",
    "run_benchmarks": "true",
    "run_tests": "false",
}
SCHEDULE_REQUEST = {**DEFAULT_REQUEST, "run_tests": "true"}


def parse_command(body: str) -> dict[str, str]:
    if "\n" in body or "\r" in body:
        raise ValueError("The /ci-rocm command must be a single line.")
    tokens = shlex.split(body)
    if not tokens or tokens[0] != "/ci-rocm":
        raise ValueError("Expected /ci-rocm.")
    request = dict(DEFAULT_REQUEST)
    seen = set()
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator or key not in request or key in seen:
            raise ValueError(f"Unknown, duplicate, or malformed input: {token!r}")
        if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"Input {key!r} must be a non-empty single-line value.")
        if key in {"downstream_configs", "run_benchmarks", "run_tests"} and value not in {"true", "false"}:
            raise ValueError(f"{key} must be true or false.")
        request[key] = value
        seen.add(key)
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()
    if args.schedule:
        request = dict(SCHEDULE_REQUEST)
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        try:
            request = parse_command(event["comment"]["body"])
        except ValueError as error:
            raise SystemExit(f"::error::{error}") from error

    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        for key, value in request.items():
            output.write(f"{key.upper()}={value}\n")


if __name__ == "__main__":
    main()
