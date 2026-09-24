"""Paired, cluster-bootstrapped metric comparison.

Why clusters: utterances synthesized from the same prompt voice share its failure modes (an accent the ASR
judge mishears, a speaking rate the duration rule gets wrong, a noisy prompt), so they are not independent
draws. Resampling utterances as if they were understates the uncertainty; resampling whole clusters (speakers
or prompts) with replacement keeps that correlation inside every resample (block/cluster bootstrap, see Liu &
Peng, arXiv:1912.09508). On the published Freya-TR-Eval baseline (495 sentences, 24 prompts from 10 held-out
speakers, corpus WER 4.32 %) the 95 % interval is [3.59, 5.09] with utterance resampling but [3.07, 5.62] with
speaker clusters -- 40-70 % wider depending on the cluster level.

Rates: corpus WER/CER = sum of edits / sum of reference words (chars), so long sentences weigh more, as in the
literature. Mean metrics (SIM, DNSMOS, ...) are the ratio with denominator 1 per utterance, so one vectorized
engine covers all.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .data import jsonl
from .metrics import summarize

METRICS = ("wer", "cer", "dnsmos_ovrl", "speaker_similarity")  # checkpoint-promotion gate of `compare`
DEFAULT_SAMPLES = 5000


class Metric(NamedTuple):
    """How one metric is aggregated and shown.

    kind "ratio": sum(fields[0]) / sum(fields[1]) over utterances (corpus rate).
    kind "rate": mean over utterances of fields[0] / fields[1], or of fields[2] when the counts are missing.
    kind "mean": mean over utterances of the first field alias present anywhere in the compared results.
    """

    label: str
    kind: str
    fields: tuple
    better: str | None  # "lower", "higher", or None when neither direction is a gain
    scale: float = 1.0  # display multiplier (100 for fractions shown in %)
    digits: int = 3


METRIC_SPECS = {
    "cer": Metric("CER %", "ratio", ("char_edits", "chars"), "lower", 100, 2),
    "wer": Metric("WER %", "ratio", ("word_edits", "words"), "lower", 100, 2),
    "wer_mean": Metric("WER % (utt. mean)", "rate", ("word_edits", "words", "wer"), "lower", 100, 2),
    "cer_mean": Metric("CER % (utt. mean)", "rate", ("char_edits", "chars", "cer"), "lower", 100, 2),
    "substitutions": Metric("S %", "ratio", ("word_substitutions", "words"), "lower", 100, 2),
    "deletions": Metric("D %", "ratio", ("word_deletions", "words"), "lower", 100, 2),
    "insertions": Metric("I %", "ratio", ("word_insertions", "words"), "lower", 100, 2),
    "speaker_similarity": Metric("SIM", "mean", ("speaker_similarity",), "higher", 1, 4),
    "sim_o": Metric("SIM-o", "mean", ("sim_o",), "higher", 1, 4),
    "dnsmos_ovrl": Metric("DNSMOS OVRL", "mean", ("dnsmos_ovrl",), "higher"),
    "dnsmos_sig": Metric("DNSMOS SIG", "mean", ("dnsmos_sig",), "higher"),
    "dnsmos_bak": Metric("DNSMOS BAK", "mean", ("dnsmos_bak",), "higher"),
    "utmos": Metric("UTMOS", "mean", ("utmos",), "higher"),
    # Evaluator.score writes clipped_fraction (16 kHz); eval_sentences.py adds clip_fraction (full band).
    "clipped_fraction": Metric("clipped %", "mean", ("clipped_fraction", "clip_fraction"), "lower", 100, 3),
    "rtf": Metric("RTF", "mean", ("rtf",), "lower"),
}


def _number(value):
    """Finite float, or None for missing/null/NaN/non-numeric values (rescored rows may carry rtf: null)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _hashable(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def cluster_bootstrap(numerators, denominators, clusters, samples=DEFAULT_SAMPLES, seed=0):
    """Bootstrap distribution of ratio statistics sum(numerator) / sum(denominator) under cluster resampling.

    numerators/denominators: [items, stats] (a corpus WER is edits over words, a mean is value over 1, a
    missing value 0 over 0). clusters: one hashable label per item. Every draw picks as many clusters as
    there are, with replacement, and sums their members, so a draw is a count vector over clusters: all
    statistics -- and both systems of a paired comparison when they are columns of one call -- see identical
    resamples. Vectorized as [B, C] counts times [C, S] cluster sums: 495 items x 5000 draws take a few ms.
    Returns (estimate [S], draws [B, S]); a draw whose resampled denominator is 0 is NaN.
    """
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if numerators.ndim == 1:
        numerators, denominators = numerators[:, None], denominators[:, None]
    if numerators.shape != denominators.shape or numerators.shape[0] != len(clusters) or not len(clusters):
        raise ValueError("Bootstrap needs one numerator/denominator row and one cluster label per item")
    if samples < 1:
        raise ValueError("Bootstrap needs at least one resample")
    codes = {}
    index = np.array([codes.setdefault(_hashable(c), len(codes)) for c in clusters])
    count = len(codes)
    num = np.zeros((count, numerators.shape[1]))
    den = np.zeros_like(num)
    np.add.at(num, index, numerators)
    np.add.at(den, index, denominators)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, numerators.shape[1]))
    chunk = max(1, 2**21 // count)  # bounds the [chunk, C] count matrix of utterance-level bootstraps
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, samples, chunk):
            size = min(chunk, samples - start)
            picks = rng.integers(0, count, size=(size, count)) + count * np.arange(size)[:, None]
            weights = np.bincount(picks.ravel(), minlength=size * count).reshape(size, count).astype(float)
            draws[start : start + size] = (weights @ num) / (weights @ den)
        estimate = num.sum(0) / den.sum(0)
    return estimate, draws


def _row_value(row, spec, fields):
    if spec.kind == "mean":
        value = _number(row.get(fields[0]))
        return None if value is None else (value, 1.0)
    numerator, denominator = _number(row.get(fields[0])), _number(row.get(fields[1]))
    counted = numerator is not None and denominator is not None and denominator > 0
    if spec.kind == "ratio":
        return (numerator, denominator) if counted else None
    if counted:
        return numerator / denominator, 1.0
    value = _number(row.get(fields[2]))
    return None if value is None else (value, 1.0)


def _column(items, ids, name, fields):
    """Per-item numerator/denominator of one metric, averaged over the item's seed replicates."""
    spec = METRIC_SPECS[name]
    num, den = np.zeros(len(ids)), np.zeros(len(ids))
    for i, item in enumerate(ids):
        values = [v for v in (_row_value(row, spec, fields) for row in items[item]) if v is not None]
        if values:
            num[i], den[i] = np.mean(values, axis=0)
    return num, den


def paired_comparison(before, after, samples=2000, seed=42):
    """Strict before/after gate for checkpoint promotion on frozen `make-cases` evaluations.

    Every row must carry all four finite gate metrics, the utterance/reference/seed keys, texts, evaluator
    identity and frozen case digest must match exactly, and failures abort. Both checkpoints are resampled
    with identical speaker clusters in `cluster_bootstrap`, so the intervals are those of paired differences.
    """

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
    for key in left:
        if left[key]["text"] != right[key]["text"]:
            raise ValueError("Reference text changed between evaluations")
        if left[key].get("evaluator") != right[key].get("evaluator"):
            raise ValueError(
                "Evaluator identity/normalization changed; rescore both checkpoints consistently"
            )
        if left[key].get("cases_sha256") != right[key].get("cases_sha256"):
            raise ValueError("Frozen evaluation cases changed")
    keys = list(left)
    speakers = [left[key]["speaker"] for key in keys]
    if len(set(speakers)) < 2:
        raise ValueError("At least two held-out speakers are needed for clustered uncertainty estimates")
    columns = [
        _column({key: [rows[key]] for key in keys}, keys, name, METRIC_SPECS[name].fields)
        for rows in (left, right)
        for name in METRICS
    ]
    num, den = np.stack([c[0] for c in columns], 1), np.stack([c[1] for c in columns], 1)
    _, draws = cluster_bootstrap(num, den, speakers, samples, seed)
    changes = draws[:, len(METRICS) :] - draws[:, : len(METRICS)]
    a, b = summarize(before), summarize(after)
    result = {
        "before": a,
        "after": b,
        "speakers": len(set(speakers)),
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
