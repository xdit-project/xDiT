"""Report which registered runners have their weights in the local HF hub cache.

Written to tell "this case fails" apart from "these weights were never on this disk",
which the GPU validation matrix cannot express on its own.

A runner's repo is not always ``settings.model_name``: several rewrite it in ``__init__``
from CLI flags (Qwen-Image switches to Qwen-Image-2512, LingBot dense to its own 1.3b
checkpoint) and the distilled Wan runners read a class-level ``_BASE_MODEL``. Resolving
that properly needs a parsed config, so every repo literal in the class source is treated
as a candidate and the runner counts as cached when any candidate is present. That can
call a runner cached on the strength of a sibling variant's weights, so it is a floor on
what is missing rather than a promise that a given invocation will load.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import re
import subprocess


def cached_repos(hub: str) -> dict[str, str]:
    repos = {}
    for entry in os.listdir(hub):
        if entry.startswith("models--"):
            repos[entry[len("models--"):].replace("--", "/")] = entry
    return repos


def candidate_repos(cls) -> list[str]:
    """Every repo id the class could resolve to, from settings and its own source."""
    names = []
    declared = getattr(cls.settings, "model_name", None)
    if declared:
        names.append(declared)
    base = getattr(cls, "_BASE_MODEL", None)
    if base:
        names.append(base)
    try:
        tree = ast.parse(inspect.cleandoc(inspect.getsource(cls)))
    except (OSError, TypeError, SyntaxError):
        return list(dict.fromkeys(names))
    for node in ast.walk(tree):
        # model_name="org/repo" as a keyword to ModelSettings, or assigned to
        # self.settings.model_name / _BASE_MODEL.
        if isinstance(node, ast.keyword) and node.arg == "model_name":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                names.append(node.value.value)
        elif isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                continue
            for target in node.targets:
                attr = getattr(target, "attr", None) or getattr(target, "id", None)
                if attr in {"model_name", "_BASE_MODEL"}:
                    names.append(node.value.value)
    return [n for n in dict.fromkeys(names) if "/" in n]


def rank_for_alias(candidates: list[str], alias: str) -> list[str]:
    """Put the repo that best matches the registered name first.

    A subclass inherits its parent's ``settings``, so the parent's repo would otherwise be
    reported for runners like the LingBot dense variant that pick their own in ``__init__``.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", alias.lower()) if t]
    squashed = re.sub(r"[^a-z0-9]", "", alias.lower())

    def score(name: str) -> int:
        lowered = name.lower()
        matched = sum(1 for token in tokens if token in lowered)
        # Tokenising "Wan2.2-I2V" yields a bare "2" that matches the 2.1 repo just as
        # well, so an unpunctuated match of the whole alias outranks any token count.
        if squashed in re.sub(r"[^a-z0-9]", "", lowered):
            matched += 10
        return matched

    return sorted(candidates, key=score, reverse=True)


def directory_size(path: str) -> str:
    result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    out = result.stdout.split()
    return out[0] if out else "?"


def default_hub() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return os.path.join(home, "hub")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", default=default_hub())
    args = parser.parse_args()

    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    repos = cached_repos(args.hub)
    by_class: dict[str, tuple[str, type]] = {}
    for alias, cls in MODEL_REGISTRY.items():
        by_class.setdefault(cls.__name__, (alias, cls))

    blocked = []
    for _, (alias, cls) in sorted(by_class.items()):
        candidates = rank_for_alias(candidate_repos(cls), alias)
        found = None
        for name in candidates:
            entry = repos.get(name)
            if entry is None:
                continue
            size = directory_size(os.path.join(args.hub, entry))
            # A metadata-only fetch is present in the cache but carries no weights.
            if size.endswith(("K", "M")):
                continue
            found = (name, size)
            break
        if found:
            print(f"{alias:28} {found[0]:52} {found[1]:>6}")
        else:
            shown = candidates[0] if candidates else "(no repo literal found)"
            extra = f" (+{len(candidates) - 1} variants)" if len(candidates) > 1 else ""
            print(f"{alias:28} {shown:52} {'--':>6}  BLOCKED{extra}")
            blocked.append((alias, candidates))

    print(f"\n{len(by_class)} runners, {len(blocked)} without usable weights:")
    for alias, candidates in blocked:
        print(f"  {alias}: {', '.join(candidates) or 'no repo literal in source'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
