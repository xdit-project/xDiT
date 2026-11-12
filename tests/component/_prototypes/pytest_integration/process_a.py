"""Mock of process A. Usage: process_a.py <case id>...

Prints its cases, writes them as a JSON array to ./a_result.json, and exits.
Environment: FAIL_PROCESS=A makes it exit with status 1 before it writes the result file.
"""

import json
import os
import sys


def main(case_ids: list[str]) -> int:
    cases = [f"A{case_id}" for case_id in case_ids]
    for case in cases:
        print(case)
    if "2" in case_ids:
        print("Case 2 chosen", file=sys.stderr)
    if os.environ.get("FAIL_PROCESS") == "A":
        return 1
    with open("a_result.json", "w") as result_file:
        json.dump(cases, result_file)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
