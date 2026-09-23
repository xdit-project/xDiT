#!/usr/bin/env python3
"""List live AITER compatibility shims and how old they are.

Shims accumulate because nobody can tell when one became safe to delete. Each
live shim carries a marker naming the date it was introduced:

    # aiter-shim: added YYYY-MM-DD. <why it exists>

A shim introduced on date D exists to support AITER builds from before roughly
D, so once the supported window has moved past D it is dead code. Historical
notes about shims already removed use a different prefix and are ignored here:

    # aiter-shim cut YYYY-MM: <what it was>

    python tests/attention/shim_report.py [--months 3]
"""

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "xfuser" / "core" / "attention"
LIVE = re.compile(r"aiter-shim:\s*added\s*(\d{4}-\d{2}-\d{2})")
LIVE_ANY = re.compile(r"aiter-shim:(?!\s*added\s*\d{4}-\d{2}-\d{2})")


def find_live(root: Path = ROOT):
    """(path, line number, date or None, text) for every live shim marker."""
    out = []
    for path in sorted(root.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "aiter-shim:" not in line:
                continue
            match = LIVE.search(line)
            date = (
                dt.date.fromisoformat(match.group(1)) if match else None
            )
            out.append((path, number, date, line.strip().lstrip("# ")))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=3,
                        help="flag shims older than this many months")
    args = parser.parse_args(argv)

    cutoff = dt.date.today() - dt.timedelta(days=30 * args.months)
    shims = find_live()
    if not shims:
        print("no live shims")
        return 0

    undated = [s for s in shims if s[2] is None]
    stale = sorted((s for s in shims if s[2] and s[2] < cutoff), key=lambda s: s[2])
    fresh = sorted((s for s in shims if s[2] and s[2] >= cutoff), key=lambda s: s[2])

    for label, group in (("STALE (older than cutoff)", stale),
                         ("current", fresh),
                         ("UNDATED", undated)):
        if not group:
            continue
        print(f"\n{label}:")
        for path, number, date, text in group:
            when = date.isoformat() if date else "????-??-??"
            rel = path.relative_to(ROOT.parents[2])
            print(f"  {when}  {rel}:{number}")
            print(f"              {text[:96]}")

    print(f"\ncutoff {cutoff.isoformat()} ({args.months} months); "
          f"{len(stale)} stale, {len(fresh)} current, {len(undated)} undated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
