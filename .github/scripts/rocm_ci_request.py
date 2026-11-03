"""Parse /ci-rocm comments without evaluating their contents."""

import json
import os
from pathlib import Path
import shlex


def parse_command(body: str) -> dict[str, str]:
    if "\n" in body or "\r" in body:
        raise ValueError("The /ci-rocm command must be a single line.")
    tokens = shlex.split(body)
    if not tokens or tokens[0] != "/ci-rocm":
        raise ValueError("Expected /ci-rocm.")
    request = {
        "benchmarks": "--tag nightly",
        "architectures": "gfx942,gfx950",
        "amd_ref": "main",
    }
    seen = set()
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator or key not in request or key in seen:
            raise ValueError(f"Unknown, duplicate, or malformed input: {token!r}")
        if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"Input {key!r} must be a non-empty single-line value.")
        request[key] = value
        seen.add(key)
    return request


def main() -> None:
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
