"""Print the host-memory shape of one sampled validation run."""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
print(f"samples: {len(rows)}  keys: {list(rows[0].keys())}")
t0 = rows[0].get("t", 0)
step = max(1, len(rows) // 24)
for row in rows[::step]:
    print(
        f"t+{row.get('t', 0) - t0:6.1f}s  "
        f"anon {row.get('anon', 0) / 1e9:7.2f} GB  "
        f"file {row.get('file', 0) / 1e9:7.2f} GB  "
        f"cur {row.get('current', 0) / 1e9:7.2f} GB"
    )
peak = max(rows, key=lambda r: r.get("anon", 0))
print(f"peak anon {peak.get('anon', 0) / 1e9:.2f} GB at t+{peak.get('t', 0) - t0:.1f}s")
