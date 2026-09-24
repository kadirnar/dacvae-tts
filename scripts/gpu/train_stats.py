"""Training-side numbers of A/B runs from their train.jsonl logs, as a tab-separated table.

  python scripts/gpu/train_stats.py /workspace/runs/ab-* > training.tsv

Per run: the last logged update, the median seconds per update (log records after update 1000, so compilation and
warm-up are excluded), the peak CUDA memory and the last validation flow. Step time is only comparable between
runs that shared a GPU model and did not compete for CPU cores.
"""

import json
import statistics
import sys
from pathlib import Path


def stats(run):
    path = Path(run) / "train.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    logs = [r for r in rows if "elapsed_seconds" in r]
    per_update = []
    for before, after in zip(logs, logs[1:]):
        if before["step"] >= 1000 and after["step"] > before["step"]:
            per_update.append(after["elapsed_seconds"] / (after["step"] - before["step"]))
    validation = [r["validation_flow"] for r in rows if "validation_flow" in r]
    return {
        "step": max((r.get("step", 0) for r in rows), default=0),
        "seconds_per_update": statistics.median(per_update) if per_update else None,
        "peak_cuda_gb": max((r.get("peak_cuda_gb", 0.0) for r in logs), default=None),
        "validation_flow": validation[-1] if validation else None,
    }


def main(runs):
    columns = ("step", "seconds_per_update", "peak_cuda_gb", "validation_flow")
    print("\t".join(["run", *columns]))
    for run in sorted(runs):
        row = stats(run)
        if row is None:
            continue
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c] if row[c] is not None else "")
                 for c in columns]
        print("\t".join([Path(run).name, *cells]))


if __name__ == "__main__":
    main(sys.argv[1:])
