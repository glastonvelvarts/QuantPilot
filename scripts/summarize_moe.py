"""The llama-bench runs of scripts/gpu_check.py moe as one table. Writes summary.md next to them.

uv run python scripts/summarize_moe.py results/<machine>
"""

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) / "moe"

lines = [
    "| model | flags | QuantPilot plans | prompt tok/s (512) | generate tok/s (128) |",
    "|---|---|---|---|---|",
]
for run in json.loads((OUT / "bench.json").read_text(encoding="utf-8")):
    planned = f"--n-cpu-moe {run['planned_n_cpu_moe']}"
    if "error" in run:
        lines.append(
            f"| {run['model']} | {run['flags']} | {planned} | error: {run['error'][-80:]} | |"
        )
        continue
    speed = {
        "pp" if row["n_prompt"] else "tg": f"{row['avg_ts']:.1f} ± {row['stddev_ts']:.1f}"
        for row in run["rows"]
    }
    lines.append(
        f"| {run['model']} | {run['flags']} | {planned} | {speed.get('pp', '-')} "
        f"| {speed.get('tg', '-')} |"
    )

summary = "\n".join(lines) + "\n"
(OUT / "summary.md").write_text(summary, encoding="utf-8")
print(summary)
