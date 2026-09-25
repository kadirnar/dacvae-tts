"""Round summary of the tr-combined A/B arms: paired comparisons against the baseline and a training table.

For every arm with a final evaluation (OUT/<arm>/step-<AB_STOP>[, -s1000]), compare_evals.py pairs it with the
baseline (both sampling seeds pooled as replicates, speaker-clustered jackknife-t over the 48 CV voices). The training
table adds what the evaluation cannot show: final validation flow, seconds per update, target frames per update (cross
prompts and tail silence spend budget on other frames) and the quick-evaluation trajectory.

  python scripts/trc/compare_arms.py --baseline base-s42 --output outputs/trc/compare-round1.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def final_dirs(out, arm, stop):
    first = out / arm / f"step-{stop:07d}"
    second = out / arm / f"step-{stop:07d}-s1000"
    dirs = [d for d in (first, second) if (d / "results.jsonl").exists()]
    return dirs


def training_row(runs, arm):
    records = [json.loads(line) for line in (runs / f"trc-{arm}" / "train.jsonl").read_text().splitlines() if line]
    steps = [r for r in records if "flow" in r and "elapsed_seconds" in r and r["step"] > 1]
    validation = [r for r in records if "validation_flow" in r]
    seconds = np.median([r["elapsed_seconds"] / 100 for r in steps]) if steps else float("nan")
    frames = np.mean([r["valid_target_frames"] for r in steps if "valid_target_frames" in r]) if steps else float("nan")
    return (validation[-1]["validation_flow"] if validation else float("nan")), seconds, frames


def quick_trajectory(out, arm):
    cells = []
    for step in (5000, 10000, 15000):
        summary = out / arm / f"step-{step:07d}" / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            cells.append(f"{100 * s['wer']:.1f}/{100 * s['cer']:.1f}")
        else:
            cells.append("")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="base-s42")
    parser.add_argument("--out", default="/workspace/outputs/trc")
    parser.add_argument("--runs", default="/workspace/runs")
    parser.add_argument("--stop", type=int, default=20000)
    parser.add_argument("--output", required=True)
    parser.add_argument("arms", nargs="*", help="Default: every arm with a final evaluation")
    args = parser.parse_args()
    out, runs = Path(args.out), Path(args.runs)
    arms = args.arms or sorted(p.name for p in out.iterdir() if p.is_dir() and final_dirs(out, p.name, args.stop))
    if args.baseline not in arms:
        sys.exit(f"baseline {args.baseline} has no final evaluation yet")
    arms = [args.baseline] + [a for a in arms if a != args.baseline]
    specs = [f"{arm}={','.join(str(d) for d in final_dirs(out, arm, args.stop))}" for arm in arms]
    report = subprocess.run([str(REPO / ".venv/bin/python"), "scripts/compare_evals.py", *specs, "--baseline",
                             args.baseline, "--stratify", "length", "--output", str(Path(args.output).with_suffix(".json"))],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    lines = [f"# tr-combined A/B arms vs `{args.baseline}` ({args.stop} updates, seeds 42 + 1000 pooled)", "",
             "## Training", "", "| arm | final val flow | s/update | target frames/update | quick WER/CER % 5k | 10k | 15k |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        try:
            flow, seconds, frames = training_row(runs, arm)
        except FileNotFoundError:
            flow = seconds = frames = float("nan")
        lines.append(f"| {arm} | {flow:.4f} | {seconds:.3f} | {frames:.0f} | " + " | ".join(quick_trajectory(out, arm)) + " |")
    lines += ["", "s/update comes from the training logs, where evaluations and store builds shared the GPU; use the #7 "
              "benchmark for speed. Target frames exclude prompt frames (cross prompts spend budget on prompts).", "",
              "## Evaluation (Freya-TR-Eval x 48 Common Voice voices)", "", report]
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
