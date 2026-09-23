"""Compare Freya runs by prompt speaking-rate group (rates from the baseline run's outputs)."""
import collections
import json
import sys

import numpy as np

from dacvae_tts.text import normalize


def load(folder):
    rows = [json.loads(l) for l in open(f"/workspace/outputs/{folder}/results.jsonl") if l.strip()]
    return {r["id"]: r for r in rows if "wer" in r}


def group_rates(base):
    by = collections.defaultdict(list)
    for r in base.values():
        by[r["prompt"]].append(r)
    return {p: np.median([len(normalize(r["text"], "turkish-v1")) / r["audio_seconds"] for r in rs]) for p, rs in by.items()}


def main():
    base_name, others = sys.argv[1], sys.argv[2:]
    base = load(base_name)
    rate = group_rates(base)
    groups = (("yavaş <13", lambda p: rate[p] < 13), ("normal 13-17", lambda p: 13 <= rate[p] <= 17), ("hızlı >17", lambda p: rate[p] > 17))
    for name in [base_name] + others:
        rows = load(name)
        line = f"{name:28s}"
        for label, cond in groups:
            rs = [r for r in rows.values() if cond(r["prompt"])]
            line += f" | {label}: WER {sum(r['word_edits'] for r in rs) / sum(r['words'] for r in rs):.3f}"
        rs = list(rows.values())
        line += f" | all: WER {sum(r['word_edits'] for r in rs) / sum(r['words'] for r in rs):.4f} CER {sum(r['char_edits'] for r in rs) / sum(r['chars'] for r in rs):.4f} OVRL {np.mean([r['dnsmos_ovrl'] for r in rs]):.3f} len {np.mean([r['audio_seconds'] for r in rs]):.2f}s"
        print(line)


if __name__ == "__main__":
    main()
