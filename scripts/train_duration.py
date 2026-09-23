"""Fit the duration predictor on same-speaker (prompt, target) pairs of a latent cache and compare it with the rules.

Every target utterance of the train split is paired with other utterances of the same speaker as voice prompts
(the inference situation: a prompt of one sentence, a new sentence to speak). The predictor (dacvae_tts/duration.py)
is a ridge regression on log target frames; the report compares its error with the byte rule used so far, the
syllable rule and the fast-prompt clamp on the validation split (unseen speakers).

  python scripts/train_duration.py --cache /workspace/data/tr55/clean --output src/dacvae_tts/duration_tr.json
"""

import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from dacvae_tts.duration import FEATURES, DurationPredictor, clamp_scale, fit, rule_frames, speaking_rate


def pairs(cache, split, per_target, seed, min_prompt=50, max_prompt=375):
    db = sqlite3.connect(f"file:{Path(cache) / 'index.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT uid, speaker, text, frames FROM samples WHERE split=?", (split,)).fetchall()
    by_speaker = defaultdict(list)
    for uid, speaker, text, frames in rows:
        by_speaker[speaker].append((uid, text, frames))
    rng = random.Random(seed)
    out = []
    for items in by_speaker.values():
        prompts = [r for r in items if min_prompt <= r[2] <= max_prompt]
        for uid, text, frames in items:
            candidates = [p for p in prompts if p[0] != uid]
            for prompt in rng.sample(candidates, min(per_target, len(candidates))):
                out.append((prompt[2], prompt[1], text, frames))
    return out


def report(name, predicted, rows):
    truth = np.array([r[3] for r in rows], dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    log_error = np.log(predicted / truth)
    relative = np.abs(predicted / truth - 1)
    rates = np.array([speaking_rate(r[0], r[1]) for r in rows])
    result = {
        "method": name,
        "mae_log": float(np.mean(np.abs(log_error))),
        "bias_log": float(np.mean(log_error)),
        "median_relative_error": float(np.median(relative)),
        "within_10pct": float(np.mean(relative <= 0.10)),
        "within_20pct": float(np.mean(relative <= 0.20)),
    }
    for label, mask in (("fast_prompts", rates > 17), ("normal_prompts", (rates >= 13) & (rates <= 17)), ("slow_prompts", rates < 13)):
        if mask.any():
            result[f"{label}_mae_log"] = float(np.mean(np.abs(log_error[mask])))
            result[f"{label}_bias_log"] = float(np.mean(log_error[mask]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-target", type=int, default=2, help="Prompts per target utterance (train split)")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train = pairs(args.cache, "train", args.per_target, args.seed)
    val = pairs(args.cache, "val", 3, args.seed + 1)
    predictor = fit(train, args.ridge)
    results = [
        report("rule_bytes", [rule_frames(r[0], r[1], r[2], "bytes") for r in val], val),
        report("rule_syllables", [rule_frames(r[0], r[1], r[2], "syllables") for r in val], val),
        report("rule_bytes_clamp", [rule_frames(r[0], r[1], r[2], "bytes") * clamp_scale(r[0], r[1]) for r in val], val),
        report("predictor", [predictor.predict(r[0], r[1], r[2]) for r in val], val),
    ]
    train_fit = report("predictor_train", [predictor.predict(r[0], r[1], r[2]) for r in train[:20000]], train[:20000])
    predictor.metadata.update(
        cache=str(Path(args.cache).resolve()),
        train_pairs=len(train),
        validation_pairs=len(val),
        validation=results,
        train_fit=train_fit,
        weights_by_feature=dict(zip(FEATURES, predictor.weights)),
        frame_rate=25,
        note="log(target frames) ridge regression; see dacvae_tts/duration.py",
    )
    predictor.save(args.output)
    print(json.dumps({"train_pairs": len(train), "validation_pairs": len(val)}, indent=2))
    for r in results + [train_fit]:
        print(json.dumps(r))
    print(json.dumps(dict(zip(FEATURES, [round(w, 4) for w in predictor.weights]))))
    assert DurationPredictor.load(args.output).weights == predictor.weights
    assert all(math.isfinite(w) for w in predictor.weights)


if __name__ == "__main__":
    main()
