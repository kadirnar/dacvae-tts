"""Coverage and false-decision rates of the evaluation intervals, by simulation with a known truth (issue #4).

The data-generating model is calibrated on a real evaluation (by default the published Freya-TR-Eval baseline:
495 sentences with their real word counts and prompt speakers): errors_i ~ Binomial(words_i,
sigmoid(mu + a_speaker + b_sentence + e_i)) with speaker effects a ~ N(0, s_spk^2), sentence difficulty b ~
N(0, s_sent^2) shared by both systems of a paired comparison, and generation noise e ~ N(0, s_noise^2) drawn per
system. System B adds delta + d_speaker to the logit, d ~ N(0, s_int^2) being a speaker x system interaction (a
setting that helps some voices and hurts others, like a duration rule on fast vs slow speakers). The estimand is the
super-population corpus WER (new voices, new sentences), computed by Monte Carlo.

Intervals (95 %): `utt` utterance-level percentile bootstrap, `clu` speaker-cluster percentile bootstrap (the
previous default of dacvae_tts.comparison), `jk-t` delete-one-speaker jackknife with t(G-1) quantiles (the default).
Reported per scenario: coverage of the single-system WER, coverage of the paired difference, and the rate at which
the paired interval excludes 0 (the false win/loss rate when the true difference is 0, the power otherwise).

  python scripts/simulate_interval_coverage.py --results outputs/freya-tr-w512-clean-60000/results.jsonl \
      --replicates 500 --output outputs/interval-coverage.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dacvae_tts.comparison import _interval, _jackknife_interval, cluster_bootstrap, cluster_jackknife

METHODS = ("utt", "clu", "jk-t")
SCENARIOS = {  # calibrated to corpus WER ~4 %, ~75 % error-free sentences, per-speaker logit WER sd ~0.6
    "null": dict(mu=-4.5, s_spk=0.55, s_sent=1.4, s_noise=0.8, delta=0.0, s_int=0.0),
    "null, weak speaker effect": dict(mu=-4.4, s_spk=0.3, s_sent=1.4, s_noise=0.8, delta=0.0, s_int=0.0),
    "null, strong speaker effect": dict(mu=-4.7, s_spk=0.8, s_sent=1.4, s_noise=0.8, delta=0.0, s_int=0.0),
    "null, speaker x system interaction": dict(mu=-4.5, s_spk=0.55, s_sent=1.4, s_noise=0.8, delta=0.0, s_int=0.4),
    "effect -20 % relative": dict(mu=-4.5, s_spk=0.55, s_sent=1.4, s_noise=0.8, delta=-0.25, s_int=0.0),
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def truth(words, mu, s_spk, s_sent, s_noise, delta, s_int, rng, count=4_000_000):
    z = mu + rng.normal(0, s_spk, count) + rng.normal(0, s_sent, count) + rng.normal(0, s_noise, count)
    w = rng.choice(words, count)
    shifted = z + delta + rng.normal(0, s_int, count)
    return float((w * sigmoid(z)).sum() / w.sum()), float((w * sigmoid(shifted)).sum() / w.sum())


def simulate(words, speaker, replicates, seed, samples, mu, s_spk, s_sent, s_noise, delta, s_int):
    rng = np.random.default_rng(seed)
    true_a, true_b = truth(words, mu, s_spk, s_sent, s_noise, delta, s_int, rng)
    true_delta = true_b - true_a
    groups = speaker.max() + 1
    cover = {m: 0 for m in METHODS}
    cover_delta = {m: 0 for m in METHODS}
    excludes_zero = {m: 0 for m in METHODS}
    width = {m: [] for m in METHODS}
    for replicate in range(replicates):
        a = rng.normal(0, s_spk, groups)[speaker]
        d = rng.normal(0, s_int, groups)[speaker]
        b = rng.normal(0, s_sent, len(words))
        za = mu + a + b + rng.normal(0, s_noise, len(words))
        zb = mu + a + b + delta + d + rng.normal(0, s_noise, len(words))
        num = np.stack([rng.binomial(words, sigmoid(za)), rng.binomial(words, sigmoid(zb))], 1).astype(float)
        den = np.stack([words, words], 1).astype(float)
        bounds = {}
        for method, clusters in (("utt", np.arange(len(words))), ("clu", speaker)):
            _, draws = cluster_bootstrap(num, den, clusters, samples, seed=replicate)
            bounds[method] = (_interval(draws[:, :1], 0.95)[0], _interval(draws[:, 1:] - draws[:, :1], 0.95)[0])
        estimate, leave = cluster_jackknife(num, den, speaker)
        bounds["jk-t"] = (
            _jackknife_interval(estimate[:1], leave[:, :1], 0.95)[0],
            _jackknife_interval(estimate[1:] - estimate[:1], leave[:, 1:] - leave[:, :1], 0.95)[0],
        )
        for method, ((low, high), (d_low, d_high)) in bounds.items():
            cover[method] += low <= true_a <= high
            width[method].append(high - low)
            cover_delta[method] += d_low <= true_delta <= d_high
            excludes_zero[method] += not d_low <= 0 <= d_high
    return {
        "true_wer": true_a,
        "true_delta": true_delta,
        "wer_coverage": {m: cover[m] / replicates for m in METHODS},
        "wer_interval_width": {m: float(np.mean(width[m])) for m in METHODS},
        "delta_coverage": {m: cover_delta[m] / replicates for m in METHODS},
        "delta_excludes_zero": {m: excludes_zero[m] / replicates for m in METHODS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="Per-sentence rows with `words` and `speaker` (results.jsonl)")
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2000, help="Bootstrap resamples per interval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "error" not in r]
    names = sorted({r["speaker"] for r in rows})
    words = np.array([r["words"] for r in rows])
    speaker = np.array([names.index(r["speaker"]) for r in rows])
    report = {"results": args.results, "sentences": len(rows), "speakers": len(names), "replicates": args.replicates,
              "scenarios": {}}
    for name, config in SCENARIOS.items():
        result = simulate(words, speaker, args.replicates, args.seed, args.samples, **config)
        report["scenarios"][name] = {"config": config, **result}
        print(name, json.dumps(result), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
