"""Mock of process B. Usage: process_b.py <case id>...

Prints its cases, writes them as a JSON array to ./b_result.json, and exits.
Environment:
- TEST_CASE=<case id> gives that case the prefix "failed_", so its test fails.
- FAIL_PROCESS=B makes it exit with status 1 before it writes the result file.
"""

import json
import os
import sys


def main(case_ids: list[str]) -> int:
    failed_case = os.environ.get("TEST_CASE")
    cases = [("failed_" if case_id == failed_case else "") + f"B{case_id}" for case_id in case_ids]
    for case in cases:
        print(case)
    if "2" in case_ids:
        print("Case 2 chosen", file=sys.stderr)
    if os.environ.get("FAIL_PROCESS") == "B":
        return 1
    with open("b_result.json", "w") as result_file:
        json.dump(cases, result_file)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
