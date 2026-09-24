import time

import numpy as np
import pytest

from dacvae_tts.comparison import cluster_bootstrap


def test_cluster_bootstrap_resamples_whole_clusters():
    # Cluster x: 4 edits / 8 words over two items; cluster y: 0 / 8. Two clusters drawn with replacement can
    # only give xx = 0.5, xy = 0.25 or yy = 0 (with probabilities 1/4, 1/2, 1/4), never an item-level mix.
    numerators, denominators = [2, 2, 0, 0], [4, 4, 3, 5]
    estimate, draws = cluster_bootstrap(numerators, denominators, ["x", "x", "y", "y"], samples=4000, seed=0)
    assert estimate[0] == pytest.approx(0.25)
    values, counts = np.unique(draws[:, 0], return_counts=True)
    assert values.tolist() == [0.0, 0.25, 0.5]
    assert counts / counts.sum() == pytest.approx([0.25, 0.5, 0.25], abs=0.03)
    # Utterance-level resampling mixes items and therefore produces other values.
    _, utterance = cluster_bootstrap(numerators, denominators, range(4), samples=4000, seed=0)
    assert len(np.unique(utterance[:, 0])) > 3
    # A mean is a ratio with denominator 1 (0/0 marks a missing value), columns share the resamples.
    estimate, draws = cluster_bootstrap(
        [[1.0, 2.0], [3.0, 0.0]], [[1, 1], [1, 0]], ["x", "y"], samples=200, seed=0
    )
    assert estimate.tolist() == [2.0, 2.0]
    assert set(np.unique(draws[:, 0])) <= {1.0, 2.0, 3.0}
    assert np.isnan(draws[draws[:, 0] == 3.0, 1]).all()  # only cluster y drawn: no value of the second metric
    with pytest.raises(ValueError):
        cluster_bootstrap([1.0], [1.0, 2.0], ["x"])


def test_cluster_bootstrap_matches_a_naive_loop():
    # Same picks as the engine (one chunk): a draw is the ratio of sums over the picked clusters' members.
    rng = np.random.default_rng(7)
    clusters = rng.choice(["a", "b", "c", "d", "e"], size=40)
    num = rng.integers(0, 5, size=(40, 3)).astype(float)
    den = rng.integers(1, 9, size=(40, 3)).astype(float)
    estimate, draws = cluster_bootstrap(num, den, clusters, samples=300, seed=11)
    names = list(dict.fromkeys(clusters))  # codes follow first appearance
    picks = np.random.default_rng(11).integers(0, len(names), size=(300, len(names)))
    for draw, row in zip(draws, picks):
        members = np.concatenate([np.flatnonzero(clusters == names[c]) for c in row])
        assert draw == pytest.approx(num[members].sum(0) / den[members].sum(0))
    assert estimate == pytest.approx(num.sum(0) / den.sum(0))


def test_cluster_bootstrap_is_fast_enough_for_freya_scale():
    rng = np.random.default_rng(0)
    words = rng.integers(2, 15, size=495).astype(float)
    edits = rng.binomial(words.astype(int), 0.05).astype(float)
    started = time.perf_counter()
    cluster_bootstrap(np.stack([edits] * 8, 1), np.stack([words] * 8, 1), np.arange(495) % 24, samples=5000)
    cluster_bootstrap(np.stack([edits] * 8, 1), np.stack([words] * 8, 1), range(495), samples=5000)
    assert time.perf_counter() - started < 5
