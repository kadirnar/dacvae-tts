"""Paired, cluster-bootstrapped comparison of per-utterance evaluation results.

Why clusters: utterances synthesized from the same prompt voice share its failure modes (an accent the ASR
judge mishears, a speaking rate the duration rule gets wrong, a noisy prompt), so they are not independent
draws. Resampling utterances as if they were understates the uncertainty; resampling whole clusters (speakers
or prompts) with replacement keeps that correlation inside every resample (block/cluster bootstrap, see Liu &
Peng, arXiv:1912.09508). On the published Freya-TR-Eval baseline (495 sentences, 24 prompts from 10 held-out
speakers, corpus WER 4.32 %) the 95 % interval is [3.59, 5.09] with utterance resampling but [3.07, 5.62] with
speaker clusters -- 40-70 % wider depending on the cluster level -- and paired WER differences below ~0.7
points are not resolvable on that set. With so few clusters the percentile interval is itself approximate (it
tends to be too narrow); more held-out voices, not more sentences per voice, is what tightens it.

Intervals: the default is the delete-one-cluster jackknife standard error with Student t(G-1) quantiles
("jackknife-t"), the few-cluster remedy of the cluster-robust inference literature (MacKinnon, Nielsen & Webb,
arXiv:2301.04527). A simulation calibrated on that baseline (scripts/simulate_interval_coverage.py: 495 sentences
with their real word counts, 10 voices, 500 replicates per scenario) measured, when the true paired difference is 0,
false win/loss rates of 10-13 % with the cluster percentile bootstrap against the nominal 5 % and 4-6 % with the
jackknife-t; paired-difference coverage 87-90 % vs 93-96 %, single-system WER coverage 87-89 % vs 91-93 %. The
utterance-level bootstrap covers a single system's WER only 64-86 % of the time and a paired difference 85 % when the
effect differs between voices. `interval="percentile"` keeps the cluster percentile bootstrap (the previous default,
bit-identical).

Pairing: both systems of a comparison are resampled with the *same* clusters in every draw, so voice and
sentence difficulty cancel in the difference and the paired interval is much tighter than two independent
ones. A difference is a win/loss only when its interval excludes 0 (direction-aware per metric), else a tie.

Rates: corpus WER/CER = sum of edits / sum of reference words (chars), so long sentences weigh more, as in the
literature; `wer_mean`/`cer_mean` are per-utterance means that weigh every sentence equally. Mean metrics
(SIM, DNSMOS, UTMOS, ...) are the ratio with denominator 1 per utterance, so one vectorized engine covers all.

Seeds: rows of one item (pairing key + cluster) from several generation seeds or result files are averaged per
item before the bootstrap (F5-TTS reports 3-seed means). Seeds are not independent units -- the cluster still
is -- so replicates reduce the within-voice noise without inflating the apparent sample size.
"""

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .data import jsonl
from .metrics import row_identity, summarize

METRICS = ("wer", "cer", "dnsmos_ovrl", "speaker_similarity")  # checkpoint-promotion gate of `compare`
DEFAULT_SAMPLES = 5000
INTERVALS = ("jackknife-t", "percentile")
DEFAULT_INTERVAL = "jackknife-t"


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
PAIRED_METRICS = (
    "cer", "wer", "wer_mean", "speaker_similarity", "sim_o", "dnsmos_ovrl", "utmos", "clipped_fraction", "rtf"
)
STRATUM_METRICS = ("cer", "wer")
SYSTEM_COLUMNS = (  # "sdi" is the substitution/deletion/insertion breakdown in one cell
    "cer", "wer", "wer_mean", "sdi", "speaker_similarity", "sim_o", "dnsmos_ovrl", "utmos",
    "clipped_fraction", "rtf",
)
TABLE_CI = {"cer", "wer", "wer_mean", "cer_mean", "speaker_similarity", "sim_o", "dnsmos_ovrl", "utmos"}
# Auto-detected pairing keys (the first present in every successful row wins): make-cases/evaluate/run-eval
# rows, monitor.py rows, eval_sentences.py rows, then bare utterance IDs and finally the text itself.
KEY_CANDIDATES = (("uid", "reference_uid"), ("uid", "prompt_uid"), ("id",), ("uid",), ("text",))
# Speaker first: prompts of one speaker are correlated too, so it is the conservative cluster.
CLUSTER_CANDIDATES = ("speaker", "prompt_uid", "reference_uid", "prompt")
LENGTH_BUCKETS = ((5, "1-5 words"), (9, "6-9 words"), (math.inf, "10+ words"))


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


def _clean(value):
    """JSON-safe float (NaN/inf become null)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


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


def _interval(draws, level):
    """Percentile interval per column, ignoring NaN draws (resamples without any value of that metric)."""
    out = np.full((draws.shape[1], 2), np.nan)
    tail = (1 - level) / 2
    for j in range(draws.shape[1]):
        finite = draws[np.isfinite(draws[:, j]), j]
        if finite.size:
            out[j] = np.quantile(finite, [tail, 1 - tail])
    return out


def cluster_jackknife(numerators, denominators, clusters):
    """Delete-one-cluster jackknife of ratio statistics sum(numerator) / sum(denominator).

    Inputs as `cluster_bootstrap`. Returns (estimate [S], leave-one-cluster-out estimates [G, S]); a leave-one-out
    value whose remaining denominator is 0 is NaN. Paired comparisons take differences of columns, like the draws.
    """
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    if numerators.ndim == 1:
        numerators, denominators = numerators[:, None], denominators[:, None]
    if numerators.shape != denominators.shape or numerators.shape[0] != len(clusters) or not len(clusters):
        raise ValueError("Jackknife needs one numerator/denominator row and one cluster label per item")
    codes = {}
    index = np.array([codes.setdefault(_hashable(c), len(codes)) for c in clusters])
    num = np.zeros((len(codes), numerators.shape[1]))
    den = np.zeros_like(num)
    np.add.at(num, index, numerators)
    np.add.at(den, index, denominators)
    with np.errstate(divide="ignore", invalid="ignore"):
        estimate = num.sum(0) / den.sum(0)
        leave = (num.sum(0) - num) / (den.sum(0) - den)
    return estimate, leave


def _jackknife_interval(estimate, leave, level):
    """estimate +- t(G-1) x jackknife standard error per column; NaN where a leave-one-out value is undefined."""
    from scipy import stats

    count = leave.shape[0]
    if count < 2:
        return np.full((leave.shape[1], 2), np.nan)
    error = np.sqrt((count - 1) / count * np.square(leave - leave.mean(0)).sum(0))
    half = stats.t.ppf(1 - (1 - level) / 2, count - 1) * error
    return np.stack([estimate - half, estimate + half], 1)


def _intervals(num, den, clusters, samples, seed, level, method, split=None):
    """(estimate [S], intervals [S', 2]) by `method`; `split` = k gives the intervals of columns k: minus :k."""
    if method == "percentile":
        estimate, draws = cluster_bootstrap(num, den, clusters, samples, seed)
        values = draws if split is None else draws[:, split:] - draws[:, :split]
        return estimate, _interval(values, level)
    estimate, leave = cluster_jackknife(num, den, clusters)
    if split is None:
        return estimate, _jackknife_interval(estimate, leave, level)
    return estimate, _jackknife_interval(estimate[split:] - estimate[:split], leave[:, split:] - leave[:, :split], level)


def verdict(low, high, better):
    """win/loss when the paired interval excludes 0 in the metric's good/bad direction, tie when it has 0."""
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return "n/a"
    if low <= 0 <= high:
        return "tie"
    if better is None:
        return "higher" if low > 0 else "lower"
    return "win" if (low > 0) == (better == "higher") else "loss"


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


def _resolve_fields(name, rows):
    spec = METRIC_SPECS[name]
    if spec.kind != "mean":
        return spec.fields
    for alias in spec.fields:  # one alias for every system, so no comparison mixes two definitions
        if any(_number(row.get(alias)) is not None for row in rows):
            return (alias,)
    return spec.fields[:1]


def detect_key(rows):
    for candidate in KEY_CANDIDATES:
        if all(row.get(field) is not None for row in rows for field in candidate):
            return candidate
    raise ValueError(
        "Cannot detect a pairing key in the rows (expected uid+reference_uid, id or text); are these "
        "per-utterance results? Pass --key"
    )


def detect_cluster(rows):
    for field in CLUSTER_CANDIDATES:
        values = {_hashable(row.get(field)) for row in rows}
        if None not in values and len(values) >= 2:
            return field
    return None


def length_bucket(row):
    """Reference length bucket: the metric-normalized word count when scored, else the text's words."""
    words = _number(row.get("words"))
    words = words if words is not None else len(str(row.get("text", "")).split())
    return next(label for limit, label in LENGTH_BUCKETS if words <= limit)


def _stratum(row, field):
    if field == "length":
        return length_bucket(row)
    value = row.get(field)
    return "(missing)" if value is None else str(value)


def _stratum_order(field, values):
    if field == "length":
        order = [label for _, label in LENGTH_BUCKETS]
        return sorted(values, key=order.index)
    return sorted(values)


def _replicate_groups(rows):
    """Accept one list of rows or a list of row lists (one per result file / generation seed)."""
    rows = list(rows)
    if rows and isinstance(rows[0], dict):
        return [rows]
    return [list(group) for group in rows]


def _index(label, groups, key, cluster, seed_field):
    """Group successful rows into items (pairing key, cluster); rows of one item from other seeds or files are
    kept together as replicates. Failed rows (with an "error") are counted, never scored."""
    items, seen, failed, total = {}, set(), 0, 0
    required = tuple(key) + ((cluster,) if cluster else ())
    for group, rows in enumerate(groups):
        for row in rows:
            total += 1
            if "error" in row:
                failed += 1
                continue
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise ValueError(f"{label}: a row lacks pairing/cluster fields {missing} (--key/--cluster)")
            item_key = tuple(_hashable(row[field]) for field in key)
            item = (item_key, _hashable(row[cluster]) if cluster else item_key)
            replicate = (group, _hashable(row.get(seed_field)) if seed_field else None)
            if (item, replicate) in seen:
                raise ValueError(
                    f"{label}: duplicate evaluation key {dict(zip(key, item_key))}; results of several seeds "
                    f"need a '{seed_field}' field or separate files"
                )
            seen.add((item, replicate))
            items.setdefault(item, []).append(row)
    if not items:
        raise ValueError(f"{label}: no successful evaluation rows")
    return {"items": items, "failed": failed, "rows": total, "replicates": max(map(len, items.values()))}


def _summary(items, ids, names, fields, samples, seed, level, utterance_ci, method=DEFAULT_INTERVAL):
    columns = {name: _column(items, ids, name, fields[name]) for name in names}
    present = [name for name in names if columns[name][1].any()]
    clusters = [item[1] for item in ids]
    result = {"items": len(ids), "clusters": len(set(clusters)), "metrics": {}}
    if not present:
        return result
    num = np.stack([columns[name][0] for name in present], 1)
    den = np.stack([columns[name][1] for name in present], 1)
    estimate, interval = _intervals(num, den, clusters, samples, seed, level, method)
    interval = interval if result["clusters"] > 1 else None
    if utterance_ci:
        utterance = _intervals(num, den, range(len(ids)), samples, seed, level, method)[1]
    for j, name in enumerate(present):
        entry = {"value": _clean(estimate[j]), "n": int((den[:, j] > 0).sum())}
        entry["ci"] = None if interval is None else [_clean(v) for v in interval[j]]
        if utterance_ci:
            entry["ci_utterance"] = [_clean(v) for v in utterance[j]]
        result["metrics"][name] = entry
    return result


def _paired(base, other, ids, names, fields, samples, seed, level, utterance_ci, method=DEFAULT_INTERVAL):
    """Paired differences other - base over the given common items, resampled with identical clusters."""
    clusters = [item[1] for item in ids]
    present, columns = [], []
    for name in names:
        num_a, den_a = _column(base, ids, name, fields[name])
        num_b, den_b = _column(other, ids, name, fields[name])
        both = (den_a > 0) & (den_b > 0)  # compare a metric only where both systems have it
        if both.any():
            present.append(name)
            columns.append((num_a * both, den_a * both, num_b * both, den_b * both, int(both.sum())))
    result = {"items": len(ids), "clusters": len(set(clusters)), "metrics": {}}
    if not present:
        return result
    k = len(present)
    num = np.stack([c[0] for c in columns] + [c[2] for c in columns], 1)
    den = np.stack([c[1] for c in columns] + [c[3] for c in columns], 1)
    estimate, interval = _intervals(num, den, clusters, samples, seed, level, method, split=k)
    interval = interval if result["clusters"] > 1 else None
    if utterance_ci:
        utterance = _intervals(num, den, range(len(ids)), samples, seed, level, method, split=k)[1]
    for j, name in enumerate(present):
        low, high = (None, None) if interval is None else map(_clean, interval[j])
        entry = {
            "baseline": _clean(estimate[j]),
            "system": _clean(estimate[k + j]),
            "delta": _clean(estimate[k + j] - estimate[j]),
            "ci": None if interval is None else [low, high],
            "verdict": verdict(low, high, METRIC_SPECS[name].better),
            "n": columns[j][4],
        }
        if utterance_ci:
            entry["ci_utterance"] = [_clean(v) for v in utterance[j]]
        result["metrics"][name] = entry
    return result


def _common_items(label, base, other):
    common = [item for item in base if item in other]
    if not common:
        base_keys = {item[0] for item in base}
        if any(item[0] in base_keys for item in other):
            raise ValueError(
                f"{label}: the systems share pairing keys but assign them to different clusters (e.g. other "
                "prompt voices); pair only runs made with the same prompts, or pass --cluster none"
            )
        raise ValueError(f"{label}: no utterances in common with the baseline; check --key")
    for item in common:
        texts = {base[item][0].get("text"), other[item][0].get("text")} - {None}
        if len(texts) > 1:
            raise ValueError(f"{label}: reference text changed between evaluations for {item[0]}")
    return common


def _identity(run, field):
    values = {json.dumps(row.get(field), sort_keys=True) for rows in run["items"].values() for row in rows}
    return values.pop() if len(values) == 1 else f"mixed:{sorted(values)}"


def check_scorers(runs, allow_mismatch=False):
    """Refuse to compare systems scored differently; returns notes.

    Every scored row carries its scorer identity (`evaluator`, compacted with metrics.row_identity: metric
    normalization, ASR model/backend/decoding, compute type and device, speaker model, protocol options such as
    band_limit_8k). WER under turkish-v1 vs turkish-v2 text, an 8 kHz band-limited vs a full-band ASR input, or
    int8 CPU vs float16 CUDA Whisper are different measurements, so a difference between or within systems is an
    error unless `allow_mismatch` turns it into a note. Rows without an identity (older results) cannot be
    checked and are only noted.
    """
    seen = {}
    for label, run in runs.items():
        for rows in run["items"].values():
            for row in rows:
                identity = row_identity(row.get("evaluator"))
                seen.setdefault(label, set()).add(None if identity is None else json.dumps(identity, sort_keys=True))
    known = sorted({i for ids in seen.values() for i in ids} - {None})
    notes = []
    if len(known) > 1:
        decoded = [json.loads(i) for i in known]
        keys = [k for k in decoded[0] if len({json.dumps(d.get(k), sort_keys=True) for d in decoded}) > 1]
        values = {
            label: {k: sorted({json.dumps(json.loads(i).get(k), sort_keys=True) for i in ids - {None}}) for k in keys}
            for label, ids in seen.items()
        }
        detail = "; ".join(
            f"{k}: " + ", ".join(f"{label}={'|'.join(v[k])}" for label, v in values.items() if v[k]) for k in keys
        )
        message = f"the systems were scored differently ({detail})"
        if not allow_mismatch:
            raise ValueError(f"{message}; rescore them with one scorer or allow the scorer mismatch explicitly")
        notes.append(f"Scorer mismatch allowed: {message}. Differences may reflect the scorer, not the systems.")
    unknown = [label for label, ids in seen.items() if None in ids]
    if known and unknown:
        notes.append(
            f"Rows of {', '.join(f'`{label}`' for label in unknown)} carry no scorer identity (`evaluator`); "
            "they cannot be checked against the other systems' scorer."
        )
    return notes


def compare_evaluations(
    systems,
    baseline=None,
    metrics=None,
    key=None,
    cluster="auto",
    seed_field="seed",
    stratify=(),
    samples=DEFAULT_SAMPLES,
    seed=0,
    level=0.95,
    utterance_ci=False,
    allow_scorer_mismatch=False,
    interval=DEFAULT_INTERVAL,
):
    """Per-system summaries with cluster-bootstrap intervals plus paired differences against a baseline.

    systems: {label: rows}, rows being one list of per-utterance dicts or a list of such lists (one per result
    file / generation seed; replicates of an item are averaged). Rows with an "error" field count as failures.
    Metrics missing from a system (or from some of its rows) are skipped there, never imputed; unknown extra
    fields are ignored. key: pairing fields (auto: uid+reference_uid, uid+prompt_uid, id, uid, then text).
    cluster: "auto" (speaker, then prompt IDs), a field name, or None/"none" for an utterance-level bootstrap.
    metrics: restricts the paired and stratum tables (default: headline metrics / CER+WER).
    stratify: fields for per-stratum reports; "length" buckets reference word counts into 1-5, 6-9 and 10+.
    interval: "jackknife-t" (default; delete-one-cluster jackknife with t(G-1) quantiles, calibrated for few
    clusters) or "percentile" (cluster percentile bootstrap, the previous default; see the module docstring).
    Systems whose rows record different scorers are refused (`check_scorers`) unless `allow_scorer_mismatch`.
    Returns a JSON-serializable report; `markdown_report` renders it.
    """
    if not systems:
        raise ValueError("No systems to compare")
    if not 0 < level < 1:
        raise ValueError("Confidence level must be in (0, 1)")
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}")
    groups = {label: _replicate_groups(rows) for label, rows in systems.items()}
    scored = [row for gs in groups.values() for g in gs for row in g if "error" not in row]
    if not scored:
        raise ValueError("No successful evaluation rows")
    key = tuple(key) if key else detect_key(scored)
    if cluster == "auto":
        cluster = detect_cluster(scored)
    elif cluster == "none":
        cluster = None
    unknown = sorted(set(metrics or ()) - set(METRIC_SPECS))
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; choose from {sorted(METRIC_SPECS)}")
    paired_names = list(metrics) if metrics else list(PAIRED_METRICS)
    stratum_names = list(metrics) if metrics else list(STRATUM_METRICS)
    fields = {name: _resolve_fields(name, scored) for name in METRIC_SPECS}
    runs = {label: _index(label, gs, key, cluster, seed_field) for label, gs in groups.items()}
    baseline = baseline if baseline is not None else next(iter(runs))
    if baseline not in runs:
        raise ValueError(f"Baseline {baseline!r} is not one of {list(runs)}")
    options = (samples, seed, level, utterance_ci, interval)
    notes = []
    if cluster is None:
        notes.append("Utterance-level intervals: they ignore the correlation within a voice.")
    aliases = {n: fields[n][0] for n, spec in METRIC_SPECS.items() if spec.kind == "mean" and spec.fields[1:]}
    report = {
        "config": {
            "baseline": baseline,
            "key": list(key),
            "cluster": cluster,
            "seed_field": seed_field,
            "bootstrap_samples": samples,
            "seed": seed,
            "level": level,
            "interval": interval,
            "aliases": aliases,
        },
        "systems": {},
        "comparisons": {},
        "strata": {},
        "notes": notes,
    }
    for label, run in runs.items():  # system tables always carry every metric that is present
        summary = _summary(run["items"], list(run["items"]), list(METRIC_SPECS), fields, *options)
        report["systems"][label] = {
            "rows": run["rows"], "failed": run["failed"], "replicates": run["replicates"], **summary
        }
    few = min(s["clusters"] for s in report["systems"].values())
    if cluster is not None and few < 20 and interval == "percentile":
        notes.append(
            f"Only {few} `{cluster}` clusters: percentile intervals from few clusters are too narrow (paired "
            "false win/loss rate 10-13 % at nominal 5 % in simulation); prefer interval jackknife-t."
        )
    elif cluster is not None and few < 20:
        notes.append(
            f"Only {few} `{cluster}` clusters: jackknife-t intervals (t with {few - 1} degrees of freedom) are wide "
            "by design; more held-out voices, not more sentences per voice, narrow them."
        )
    notes.extend(check_scorers(runs, allow_scorer_mismatch))
    for field in ("asr_backend", "cases_sha256"):
        if len({_identity(run, field) for run in runs.values()}) > 1:
            notes.append(f"`{field}` differs between systems: a difference may reflect the scorer.")
    base = runs[baseline]["items"]
    common = {
        label: _common_items(label, base, run["items"]) for label, run in runs.items() if label != baseline
    }
    for label, ids in common.items():
        other = runs[label]["items"]
        comparison = _paired(base, other, ids, paired_names, fields, *options)
        comparison["unpaired"] = {"baseline": len(base) - len(ids), "system": len(other) - len(ids)}
        if any(comparison["unpaired"].values()):
            notes.append(
                f"`{label}`: {comparison['unpaired']['baseline']} baseline and "
                f"{comparison['unpaired']['system']} system utterances have no scored partner (failures or "
                "different sets) and are left out of the paired differences."
            )
        revoiced = sum(base[item][0].get("speaker") != other[item][0].get("speaker") for item in ids)
        if revoiced:  # e.g. --cluster prompt across prompt draws whose files are all named prompt-NN.wav
            notes.append(
                f"`{label}`: {revoiced} paired utterances have another `speaker` than in the baseline; those "
                "pairs share the text but not the voice."
            )
        report["comparisons"][label] = comparison
    for field in stratify:
        strata = {
            label: {item: _stratum(rows[0], field) for item, rows in run["items"].items()}
            for label, run in runs.items()
        }
        report["strata"][field] = {}
        for value in _stratum_order(field, {v for labels in strata.values() for v in labels.values()}):
            entry = {"systems": {}, "comparisons": {}}
            for label, run in runs.items():
                ids = [item for item in run["items"] if strata[label][item] == value]
                if ids:
                    entry["systems"][label] = _summary(run["items"], ids, stratum_names, fields, *options)
            for label, ids in common.items():
                ids = [item for item in ids if strata[baseline][item] == value]
                if ids:
                    other = runs[label]["items"]
                    entry["comparisons"][label] = _paired(base, other, ids, stratum_names, fields, *options)
            report["strata"][field][value] = entry
    return report


def _format(value, name, signed=False):
    if value is None:
        return "-"
    spec = METRIC_SPECS[name]
    return f"{value * spec.scale:{'+' if signed else ''}.{spec.digits}f}"


def _format_ci(interval, name):
    if not interval or interval[0] is None:
        return "-"
    return f"[{_format(interval[0], name)}, {_format(interval[1], name)}]"


def _with_ci(entry, name):
    if entry is None:
        return "-"
    text = _format(entry["value"], name)
    return f"{text} {_format_ci(entry['ci'], name)}" if entry.get("ci") else text


def _mark(verdict):
    return f"**{verdict}**" if verdict in ("win", "loss") else verdict


def _delta_cell(entry, name):
    if entry is None:
        return "-"
    delta = _format(entry["delta"], name, signed=True)
    return f"{delta} {_format_ci(entry['ci'], name)} {_mark(entry['verdict'])}"


def _table(header, rows, numeric_from=1):
    align = ["---" if i < numeric_from else "---:" for i in range(len(header))]
    return ["| " + " | ".join(line) + " |" for line in (header, align, *rows)]


def _system_table(systems):
    present = {name for s in systems.values() for name in s["metrics"]}
    columns = [c for c in SYSTEM_COLUMNS if c in present or (c == "sdi" and "substitutions" in present)]
    seeds = any(s["replicates"] > 1 for s in systems.values())
    utterance = any("ci_utterance" in e for s in systems.values() for e in s["metrics"].values())
    header = ["system", "n", "clusters", "failed"] + (["seeds"] if seeds else [])
    for column in columns:
        header.append("S/D/I %" if column == "sdi" else METRIC_SPECS[column].label)
        if column == "wer" and utterance:
            header.append("WER % utt.-level CI")
    rows = []
    for label, s in systems.items():
        metrics = s["metrics"]
        row = [f"`{label}`", str(s["items"]), str(s["clusters"]), str(s["failed"])]
        row += [str(s["replicates"])] if seeds else []
        for column in columns:
            if column == "sdi":
                parts = ("substitutions", "deletions", "insertions")
                row.append("/".join(_format(metrics.get(n, {}).get("value"), n) for n in parts))
                continue
            entry = metrics.get(column)
            if column in TABLE_CI:
                row.append(_with_ci(entry, column))
            else:
                row.append(_format(entry and entry["value"], column))
            if column == "wer" and utterance:
                row.append(_format_ci(entry and entry.get("ci_utterance"), column))
        rows.append(row)
    return _table(header, rows)


def _paired_table(comparisons, baseline, level):
    utterance = any("ci_utterance" in e for c in comparisons.values() for e in c["metrics"].values())
    header = ["system", "metric", "n", f"`{baseline}`", "system value", "Δ", f"{level} CI", "verdict"]
    header += ["utt.-level CI"] if utterance else []
    rows = []
    for label, comparison in comparisons.items():
        for name, entry in comparison["metrics"].items():
            row = [
                f"`{label}`",
                METRIC_SPECS[name].label,
                str(entry["n"]),
                _format(entry["baseline"], name),
                _format(entry["system"], name),
                _format(entry["delta"], name, signed=True),
                _format_ci(entry["ci"], name),
                _mark(entry["verdict"]),
            ]
            if utterance:
                row.append(_format_ci(entry.get("ci_utterance"), name))
            rows.append(row)
    return _table(header, rows, numeric_from=2)


def _stratum_table(field, strata, baseline, paired):
    shown = {name for e in strata.values() for s in e["systems"].values() for name in s["metrics"]}
    names = [name for name in METRIC_SPECS if name in shown]
    header = [field, "system", "n", "clusters"] + [METRIC_SPECS[n].label for n in names]
    header += [f"Δ {METRIC_SPECS[n].label} vs `{baseline}`" for n in names] if paired else []
    rows = []
    for value, entry in strata.items():
        for label, s in entry["systems"].items():
            row = [value, f"`{label}`", str(s["items"]), str(s["clusters"])]
            row += [_with_ci(s["metrics"].get(n), n) for n in names]
            if paired:
                comparison = entry["comparisons"].get(label)
                row += [_delta_cell(comparison and comparison["metrics"].get(n), n) for n in names]
            rows.append(row)
    return _table(header, rows, numeric_from=2)


def markdown_report(report):
    """Markdown tables: one row per system, the paired differences vs the baseline, then any strata."""
    config = report["config"]
    level = f"{config['level']:.0%}"
    unit = f"`{config['cluster']}` clusters" if config["cluster"] else "utterances (no clustering)"
    if config.get("interval", "percentile") == "percentile":
        method = f"Bootstrap over {unit}: B = {config['bootstrap_samples']}, seed {config['seed']}, {level} percentile"
    else:
        method = f"Delete-one-cluster jackknife over {unit}: {level} t"
    lines = [
        f"{method} intervals; utterances paired on ({', '.join(config['key'])}). A difference is a "
        "win/loss only when its paired interval excludes 0, otherwise a tie.",
        "",
        "### Systems",
        "",
        *_system_table(report["systems"]),
    ]
    if report["comparisons"]:
        lines += ["", f"### Paired differences vs `{config['baseline']}`", ""]
        lines += _paired_table(report["comparisons"], config["baseline"], level)
    for field, strata in report["strata"].items():
        lines += ["", f"### By `{field}`", ""]
        lines += _stratum_table(field, strata, config["baseline"], bool(report["comparisons"]))
    if report["notes"]:
        lines += ["", "Notes:", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def paired_comparison(before, after, samples=2000, seed=42, interval=DEFAULT_INTERVAL):
    """Strict before/after gate for checkpoint promotion on frozen `make-cases` evaluations.

    Unlike `compare_evaluations`, every row must carry all four finite gate metrics, the utterance/reference/
    seed keys, texts, evaluator identity and frozen case digest must match exactly, and failures abort.
    Intervals are speaker-clustered and paired, by `interval` as in `compare_evaluations`.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}")

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
    _, bounds = _intervals(num, den, speakers, samples, seed, 0.95, interval, split=len(METRICS))
    a, b = summarize(before), summarize(after)
    result = {
        "before": a,
        "after": b,
        "speakers": len(set(speakers)),
        "bootstrap_samples": samples,
        "interval": interval,
        "changes": {
            key: {"delta": b[key] - a[key], "ci95": bounds[i].tolist()} for i, key in enumerate(METRICS)
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
    result = paired_comparison(list(jsonl(args.before)), list(jsonl(args.after)), args.bootstrap, args.seed,
                               getattr(args, "interval", DEFAULT_INTERVAL))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
