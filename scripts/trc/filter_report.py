"""Yield of transcript/quality filters on a scored corpus (scripts/transcribe_corpus.py output), for issue #15.

Joins OUTPUT scores.jsonl with a merged cache index (uid -> frames, speaker, split) and optionally a program map
(speaker -> source recording), then reports kept hours, rows and speakers for a grid of CER / DNSMOS OVRL thresholds,
the score distributions, and the per-program medians (programs whose clips are mostly mismatched or noisy).

  python scripts/trc/filter_report.py --scores outputs/trc-scores/scores.jsonl --cache data/trc/merged \
      --programs data/meta/metadata.json --output outputs/trc/filter-report.md
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--programs", help="tr-combined metadata.json (speaker_id -> original_dataset)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scores = {}
    for line in open(args.scores):
        row = json.loads(line)
        scores[row["uid"]] = row
    db = sqlite3.connect(f"file:{Path(args.cache) / 'index.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT uid, frames, speaker, split, text FROM samples").fetchall()
    program = {}
    if args.programs:
        for segment in json.load(open(args.programs))["segments"]:
            program.setdefault(segment["speaker_id"], segment["original_dataset"])
    table = []
    for uid, frames, speaker, split, text in rows:
        score = scores.get(uid, {})
        table.append(dict(uid=uid, hours=frames / 25 / 3600, speaker=speaker, split=split,
                          cer=score.get("cer", np.nan), ovrl=score.get("dnsmos_ovrl", np.nan),
                          sig=score.get("dnsmos_sig", np.nan), bak=score.get("dnsmos_bak", np.nan),
                          words=len(text.split()), program=program.get(speaker, speaker)))
    cer = np.array([r["cer"] for r in table])
    ovrl = np.array([r["ovrl"] for r in table])
    hours = np.array([r["hours"] for r in table])
    words = np.array([r["words"] for r in table])
    speakers = np.array([r["speaker"] for r in table])
    lines = [f"# Filter report: {len(table)} cached rows, {hours.sum():.1f} h, {len(set(speakers))} speakers", "",
             f"Scored: {np.isfinite(cer).sum()} with CER, {np.isfinite(ovrl).sum()} with DNSMOS.", ""]
    lines += ["| quantity | p5 | p10 | p25 | p50 | p75 | p90 | p95 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in (("CER", cer), ("DNSMOS OVRL", ovrl)):
        q = np.nanpercentile(values, [5, 10, 25, 50, 75, 90, 95])
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in q) + " |")
    lines += ["", "Kept share of hours (rows, speakers) for CER <= c and OVRL >= o, >= 2 words:", "",
              "| CER <= | " + " | ".join(f"OVRL >= {o}" for o in ("-", 2.0, 2.3, 2.5, 2.8)) + " |",
              "|---:|" + "---:|" * 5]
    for c in (0.05, 0.10, 0.15, 0.20, 1.0):
        cells = []
        for o in (-1, 2.0, 2.3, 2.5, 2.8):
            keep = np.nan_to_num(cer, nan=9) <= c
            keep &= np.nan_to_num(ovrl, nan=-9) >= o
            keep &= words >= 2
            cells.append(f"{hours[keep].sum():.0f} h ({keep.sum() / len(table):.0%}, {len(set(speakers[keep]))} spk)")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    by_program = defaultdict(list)
    for r in table:
        by_program[r["program"]].append(r)
    stats = []
    for name, items in by_program.items():
        stats.append((name, len(items), sum(r["hours"] for r in items), np.nanmedian([r["cer"] for r in items]),
                      np.nanmedian([r["ovrl"] for r in items])))
    stats.sort(key=lambda s: -s[2])
    median_cer = np.array([s[3] for s in stats])
    median_ovrl = np.array([s[4] for s in stats])
    program_hours = np.array([s[2] for s in stats])
    lines += ["", f"Programs: {len(stats)}; hours in programs with median CER > 0.10: "
              f"{program_hours[median_cer > 0.10].sum():.0f} h; median OVRL < 2.3: "
              f"{program_hours[median_ovrl < 2.3].sum():.0f} h", "",
              "| program | rows | hours | median CER | median OVRL |", "|---|---:|---:|---:|---:|"]
    for name, count, h, c, o in stats[:30]:
        lines.append(f"| {name[:13]} | {count} | {h:.1f} | {c:.3f} | {o:.2f} |")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
