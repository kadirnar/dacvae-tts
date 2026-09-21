"""Illustrative padding/utilisation simulation using the repository's own sampler.

The corpus is synthetic (the real 4M-row corpus is not available): heavy-tailed
utterances per speaker, durations 1-15 s. Only relative comparisons are meaningful.
"""

import json
import random
import sys

import numpy as np

from dacvae_tts.data import BucketBatchSampler

rng = np.random.default_rng(0)
speakers = 20000
per_speaker = np.clip(rng.lognormal(2.2, 1.1, speakers).astype(int), 2, 2000)
total = int(per_speaker.sum())
seconds = np.clip(rng.gamma(2.2, 3.0, total), 1.0, 15.0)
lengths = np.round(seconds * 25).astype(np.int32)
group_start = np.repeat(np.cumsum(per_speaker) - per_speaker, per_speaker)
group_end = np.repeat(np.cumsum(per_speaker), per_speaker)
print(f"rows={total} hours={seconds.sum() / 3600:.0f} mean_seconds={seconds.mean():.2f}")

max_ref = np.empty(total, dtype=np.int32)
for start in np.unique(group_start):
    end = group_end[start]
    max_ref[start:end] = lengths[start:end].max()
pessimistic = lengths + max_ref


def reference_lengths(epoch, seed=42):
    # Same rule as LatentDataset.__getitem__.
    out = np.empty(total, dtype=np.int32)
    for index in range(total):
        r = random.Random(seed + epoch * total + index)
        start, end = int(group_start[index]), int(group_end[index])
        ref = r.randrange(start, end - 1)
        ref += ref >= index
        out[index] = lengths[ref]
    return out


refs = reference_lengths(0)
exact = lengths + refs


def evaluate(name, sort_costs, batch_size, frame_budget):
    sampler = BucketBatchSampler(sort_costs, batch_size, 0, 1, 42, frame_budget)
    batches = sampler.batches()
    padded = valid = target = 0
    sizes, tokens = [], []
    for batch in batches:
        index = np.asarray(batch)
        true_cost = exact[index]
        padded += int(true_cost.max()) * len(index)
        valid += int(true_cost.sum())
        target += int(lengths[index].sum())
        sizes.append(len(index))
        tokens.append(int(true_cost.max()) * len(index))
    result = {
        "policy": name,
        "batches": len(batches),
        "mean_batch_size": float(np.mean(sizes)),
        "padding_waste": 1 - valid / padded,
        "supervised_fraction_of_padded_compute": target / padded,
        "padded_frames_per_batch_mean": float(np.mean(tokens)),
        "padded_frames_per_batch_p5_p95": [float(np.percentile(tokens, 5)), float(np.percentile(tokens, 95))],
    }
    print(json.dumps(result))
    return result


results = [
    evaluate("current: pessimistic cost, 16 fixed", pessimistic, 16, 0),
    evaluate("exact epoch cost, 16 fixed", exact, 16, 0),
    evaluate("exact cost, frame budget 6000, cap 64", exact, 64, 6000),
    evaluate("pessimistic cost, frame budget 6000, cap 64", pessimistic, 64, 6000),
]
print("reference share of valid frames:", float(refs.sum() / exact.sum()))
json.dump(results, open(sys.argv[1] + "/sim_batching.json", "w"), indent=1)
