"""Paired, speaker-clustered metric comparison for checkpoint promotion."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import jsonl
from .metrics import summarize

METRICS = ("wer", "cer", "dnsmos_ovrl", "speaker_similarity")


def paired_comparison(before, after, samples=2000, seed=42):
    def indexed(rows):
        index = {}
        for row in rows:
            if "error" in row:
                raise ValueError("Failed generations must be investigated before checkpoint promotion")
            key = (row["uid"], row["reference_uid"], row["seed"])
            if key in index:
                raise ValueError("Duplicate evaluation key")
            if any(k not in row or not np.isfinite(row[k]) for k in METRICS):
                raise ValueError("Comparison requires all four finite metrics")
            index[key] = row
        return index

    left, right = indexed(before), indexed(after)
    if not left or left.keys() != right.keys():
        raise ValueError("Evaluations must contain identical utterances, reference IDs and seeds")
    groups = defaultdict(list)
    for key in left:
        if left[key]["text"] != right[key]["text"]:
            raise ValueError("Reference text changed between evaluations")
        if left[key].get("evaluator") != right[key].get("evaluator"):
            raise ValueError(
                "Evaluator identity/normalization changed; rescore both checkpoints consistently"
            )
        if left[key].get("cases_sha256") != right[key].get("cases_sha256"):
            raise ValueError("Frozen evaluation cases changed")
        groups[left[key]["speaker"]].append(key)
    if len(groups) < 2:
        raise ValueError("At least two held-out speakers are needed for clustered uncertainty estimates")
    names = list(groups)
    rng = np.random.default_rng(seed)
    changes = np.empty((samples, len(METRICS)))
    for index in range(samples):
        keys = [key for group in rng.choice(names, len(names), replace=True) for key in groups[group]]
        a, b = summarize([left[k] for k in keys]), summarize([right[k] for k in keys])
        changes[index] = [b[k] - a[k] for k in METRICS]
    a, b = summarize(before), summarize(after)
    result = {
        "before": a,
        "after": b,
        "speakers": len(groups),
        "bootstrap_samples": samples,
        "changes": {
            key: {"delta": b[key] - a[key], "ci95": np.quantile(changes[:, i], [0.025, 0.975]).tolist()}
            for i, key in enumerate(METRICS)
        },
    }
    # Conservative automatic gate; listening and speed checks remain separate requirements.
    result["metric_gate_passed"] = (
        all(result["changes"][k]["ci95"][1] < 0 for k in ("wer", "cer"))
        and result["changes"]["dnsmos_ovrl"]["ci95"][0] > 0
        and result["changes"]["speaker_similarity"]["ci95"][0] >= 0
    )
    return result


def compare(args):
    result = paired_comparison(list(jsonl(args.before)), list(jsonl(args.after)), args.bootstrap, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
