import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from dacvae_tts.data import LatentDataset, speaker_split
from dacvae_tts.speakers import (
    RowAudio,
    assign_split,
    average_linkage,
    cluster_labels,
    cluster_speakers,
    inconsistent_labels,
    label_centroids,
    leakage,
    load_split_map,
    read_speaker_list,
    split_group,
    split_map_from_clusters,
    threshold_components,
)

EPISODE = r"^(.+)_speaker_\d+$"


def voices(count, dim=16, seed=0):
    """Orthonormal synthetic voices: different voices have cosine 0."""
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))[0][:count]


def utterances(voice, count, rng, noise=0.15):
    return voice + noise * rng.normal(size=(count, len(voice)))


def test_split_key_extraction_and_episode_level_split():
    assert split_group("ep1_speaker_0") == "ep1_speaker_0"
    assert split_group("ep1_speaker_0", EPISODE) == "ep1"
    assert split_group("show12_speaker_3", r"^(?P<prefix>[a-z]+)(?P<key>\d+)_") == "12"
    assert split_group("show12_speaker_3", r"^[a-z]+") == "show"
    with pytest.raises(ValueError, match="does not match"):
        split_group("anonymous", EPISODE)
    with pytest.raises(ValueError, match="Invalid --split-key"):
        split_group("ep1_speaker_0", "(")
    by_episode = {}
    for episode in range(2000):
        for k in range(3):
            label = f"ep{episode}_speaker_{k}"
            by_episode.setdefault(episode, set()).add(speaker_split(split_group(label, EPISODE)))
    assert all(len(splits) == 1 for splits in by_episode.values())
    assert set().union(*by_episode.values()) == {"train", "val", "test"}
    # Merge-time precedence: label entry, then split-key entry, then split-key hash, else unchanged.
    mapping = {"ep1_speaker_0": "test", "ep2": "val"}
    assert assign_split("ep1_speaker_0", "train", 42, EPISODE, mapping) == "test"
    assert assign_split("ep2_speaker_5", "train", 42, EPISODE, mapping) == "val"
    assert assign_split("ep3_speaker_0", "val", 42, EPISODE, mapping) == speaker_split("ep3", 42)
    assert assign_split("ep3_speaker_0", "val", 42, None, mapping) == "val"
    assert assign_split("ep3_speaker_0", "val", 42) == "val"


def test_average_linkage_does_not_chain_dissimilar_voices():
    a, b, c = np.array([1.0, 0.0]), np.array([0.5**0.5, 0.5**0.5]), np.array([0.0, 1.0])
    centroids = np.stack([a, b, c])
    roots = threshold_components(centroids, 0.65)
    assert len(set(roots.tolist())) == 1  # single linkage chains a~b~c although a.c = 0
    assignment = average_linkage(centroids @ centroids.T, 0.65)
    assert assignment[0] == assignment[1] != assignment[2]
    clusters, fallbacks = cluster_labels(centroids, 0.65)
    assert clusters.tolist() == [0, 0, 1] and fallbacks == 0
    assert cluster_labels(centroids, 0.75)[0].tolist() == [0, 1, 2]
    # An oversized component keeps its single-linkage grouping and is counted.
    clusters, fallbacks = cluster_labels(centroids, 0.65, max_component=2)
    assert clusters.tolist() == [0, 0, 0] and fallbacks == 1


def synthetic_corpus():
    rng = np.random.default_rng(3)
    v = voices(6)
    layout = {  # label: (voice, split)
        "ep1_speaker_0": (0, "train"),
        "ep2_speaker_1": (0, "train"),
        "ep3_speaker_0": (0, "test"),  # the recurring host leaks into test
        "ep1_speaker_1": (1, "train"),
        "ep4_speaker_0": (2, "val"),
        "ep5_speaker_0": (2, "test"),  # one voice in two held-out labels, never trained on
        "ep4_speaker_1": (3, "test"),
        "ep9_speaker_0": (4, "train"),
    }
    embeddings, labels = [], []
    for label, (voice, _) in layout.items():
        embeddings.append(utterances(v[voice], 6, rng))
        labels += [label] * 6
    return np.concatenate(embeddings), labels, {label: split for label, (_, split) in layout.items()}


def test_clusters_leakage_and_split_map_on_synthetic_embeddings():
    embeddings, labels, split_of = synthetic_corpus()
    names, centroids, members = label_centroids(embeddings, labels)
    assert names == sorted(split_of) and all(len(m) == 6 for m in members)
    splits = [split_of[name] for name in names]
    clusters, _ = cluster_labels(centroids, 0.65)
    cluster = dict(zip(names, clusters.tolist()))
    assert cluster["ep1_speaker_0"] == cluster["ep2_speaker_1"] == cluster["ep3_speaker_0"]
    assert cluster["ep4_speaker_0"] == cluster["ep5_speaker_0"] != cluster["ep1_speaker_0"]
    assert len({cluster[n] for n in ("ep1_speaker_0", "ep1_speaker_1", "ep4_speaker_0", "ep4_speaker_1")}) == 4
    assert cluster["ep1_speaker_0"] == 0  # the largest cluster comes first
    leaked = leakage(names, centroids, splits, 0.65, clusters)
    assert list(leaked) == ["ep3_speaker_0"]
    entry = leaked["ep3_speaker_0"]
    assert entry["split"] == "test" and entry["score"] > 0.9 and entry["cluster_has_train"]
    assert entry["nearest_train_label"] in {"ep1_speaker_0", "ep2_speaker_1"}
    assert leakage(names, centroids, splits, 0.999) == {}

    rows = {"ep4_speaker_0": 10, "ep5_speaker_0": 3}
    mapping = split_map_from_clusters(names, clusters, splits, rows)
    assert mapping["ep3_speaker_0"] == "train"  # keep-train: a trained voice is never held out
    assert mapping["ep4_speaker_0"] == mapping["ep5_speaker_0"] == "val"  # more held-out rows in val
    assert mapping["ep4_speaker_1"] == "test"  # a distinct voice keeps its split without a split key
    assert {k: v for k, v in mapping.items() if v != split_of[k]} == {
        "ep3_speaker_0": "train",
        "ep5_speaker_0": "val",
    }
    by_episode = split_map_from_clusters(names, clusters, splits, rows, split_key=EPISODE)
    assert by_episode["ep4_speaker_1"] == by_episode["ep4_speaker_0"] == by_episode["ep5_speaker_0"]
    # Key entries route labels the analysis never saw (e.g. dropped singletons) to their episode's split.
    assert set(by_episode) - set(names) == {"ep1", "ep2", "ep3", "ep4", "ep5", "ep9"}
    assert all(by_episode[name] == by_episode[split_group(name, EPISODE)] for name in names)
    assert assign_split("ep4_speaker_7", "train", 42, EPISODE, by_episode) == by_episode["ep4_speaker_0"]
    hashed = split_map_from_clusters(names, clusters, splits, policy="hash", seed=7, split_key=EPISODE)
    assert hashed["ep9_speaker_0"] == speaker_split("ep9", 7)  # a single-episode voice: the split-key hash
    assert len({hashed[n] for n in ("ep1_speaker_0", "ep2_speaker_1", "ep3_speaker_0", "ep1_speaker_1")}) == 1
    with pytest.raises(ValueError, match="policy"):
        split_map_from_clusters(names, clusters, splits, policy="majority")


def test_inconsistent_labels_flag_two_voices_under_one_label():
    rng = np.random.default_rng(5)
    v = voices(3)
    embeddings = np.concatenate(
        [utterances(v[0], 4, rng), utterances(v[1], 2, rng), utterances(v[2], 6, rng), utterances(v[0], 2, rng)]
    )
    labels = ["mixed"] * 6 + ["clean"] * 6 + ["short"] * 2
    uids = [f"u{i}" for i in range(len(labels))]
    result = inconsistent_labels(embeddings, labels, uids)
    assert list(result) == ["mixed"]
    assert [o["uid"] for o in result["mixed"]["outliers"]] == ["u4", "u5"]
    assert result["mixed"]["outlier_fraction"] == pytest.approx(1 / 3, abs=1e-5)
    with pytest.raises(ValueError, match="two utterances"):
        inconsistent_labels(embeddings, labels, uids, min_utterances=1)


def test_speaker_list_and_split_map_formats(tmp_path):
    path = tmp_path / "labels"
    for text in ('["a", "b"]', '{"a": {"score": 0.9}, "b": {}}', "a\n\nb\n"):
        path.write_text(text)
        assert read_speaker_list(path) == {"a", "b"}
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON list"):
        read_speaker_list(path)
    path.write_text('{"a": "train", "ep2": "test"}')
    assert load_split_map(path) == {"a": "train", "ep2": "test"}
    for text in ('{"a": "dev"}', '["a"]'):
        path.write_text(text)
        with pytest.raises(ValueError, match="split-map"):
            load_split_map(path)


def cache_row(cache, uid):
    fields = ("uid", "speaker", "split", "audio", "shard", "offset", "frames", "samples")
    with sqlite3.connect(cache / "index.sqlite") as db:
        return dict(zip(fields, db.execute(f"SELECT {','.join(fields)} FROM samples WHERE uid=?", (uid,)).fetchone()))


def test_row_audio_prefers_original_and_decodes_raw_latents(cache, tmp_path):
    calls = []

    def decoder(z):
        calls.append(z.clone())
        return torch.sin(torch.arange(len(z) * 512) / 7.0)  # 24 kHz, hop 512

    row = cache_row(cache, "train-1-2")
    audio, kind = RowAudio(cache, decoder=decoder)(row)
    assert kind == "latents" and audio.dtype == np.float32
    assert len(audio) == -(-row["samples"] * 2 // 3)  # 24 kHz -> 16 kHz
    dataset = LatentDataset(cache, "train")
    index = next(i for i in range(len(dataset)) if dataset.row(i)["uid"] == "train-1-2")
    raw = dataset.row(index)["latents"] * dataset.std + dataset.mean  # shards hold unnormalized latents
    torch.testing.assert_close(calls[-1], raw, atol=1e-5, rtol=1e-5)
    RowAudio(cache, max_seconds=0.2, decoder=decoder)(row)
    assert len(calls[-1]) == round(0.2 * 24000 / 512) < row["frames"]

    original = tmp_path / "original.wav"
    sf.write(original, np.random.default_rng(0).normal(0, 0.1, 4000).astype(np.float32), 8000)
    audio, kind = RowAudio(cache, decoder=decoder)(dict(row, audio=str(original)))
    assert kind == "audio" and len(audio) == 8000 and len(calls) == 2
    assert RowAudio(cache, "latents", decoder=decoder)(dict(row, audio=str(original)))[1] == "latents"
    with pytest.raises(ValueError, match="Original audio missing"):
        RowAudio(cache, "audio", decoder=decoder)(row)
    with pytest.raises(ValueError, match="source"):
        RowAudio(cache, "mixed")


VOICE_OF = {  # conftest labels; val-0 is train-0's voice and test-1 is train-1's
    "train-0": 0, "train-1": 1, "train-2": 2, "train-3": 3,
    "val-0": 0, "val-1": 4, "val-2": 5, "val-3": 6,
    "test-0": 7, "test-1": 1, "test-2": 8, "test-3": 9,
}  # fmt: skip


def fake_audio(calls):
    def load(row):
        calls.append(row["uid"])
        return np.array([VOICE_OF[row["speaker"]], sum(map(ord, row["uid"]))], dtype=np.float32), "latents"

    return load


def fake_embed(audio):
    voice, seed = int(audio[0]), int(audio[1])
    return voices(10)[voice] + 0.1 * np.random.default_rng(seed).normal(size=16)


def test_cluster_speakers_end_to_end_with_cached_embeddings(cache, tmp_path):
    calls = []
    output = tmp_path / "clusters"
    args = SimpleNamespace(cache=cache, output=output, per_label=2, threshold=0.65, embedder="fake")
    summary = cluster_speakers(args, embed=fake_embed, load=fake_audio(calls))
    assert len(calls) == 24 and summary["embedded_utterances"] == 24 and summary["labels"] == 12
    clusters = json.loads((output / "clusters.json").read_text())
    assert clusters["val-0"] == clusters["train-0"] and clusters["test-1"] == clusters["train-1"]
    assert read_speaker_list(output / "leakage.json") == {"val-0", "test-1"}
    mapping = json.loads((output / "split_map.json").read_text())
    changed = {label: split for label, split in mapping.items() if split != label.split("-")[0]}
    assert changed == {"val-0": "train", "test-1": "train"}
    assert summary["held_out"] == {
        "val": {"labels": 4, "leaked": 1, "clean": 3},
        "test": {"labels": 4, "leaked": 1, "clean": 3},
    }
    assert summary["split_map"]["rows_changed"] == {"test->train": 3, "val->train": 3}
    assert json.loads((output / "inconsistent_labels.json").read_text()) == {}
    assert summary["embedding_sources"] == {"latents": 24}

    # Re-clustering at another threshold reuses every embedding: no audio is loaded.
    strict = cluster_speakers(
        SimpleNamespace(**{**vars(args), "threshold": 0.99}), embed=fake_embed, load=fake_audio(calls)
    )
    assert len(calls) == 24 and strict["clusters"] == 12
    assert json.loads((output / "leakage.json").read_text()) == {}
    # More utterances per label embed only the new rows.
    cluster_speakers(SimpleNamespace(**{**vars(args), "per_label": 3}), embed=fake_embed, load=fake_audio(calls))
    assert len(calls) == 36
    with pytest.raises(ValueError, match="holds embeddings"):
        cluster_speakers(SimpleNamespace(**{**vars(args), "embedder": "other"}), embed=fake_embed)

    # A bad clip is skipped and reported; an error on every row (codec, model) aborts early.
    def flaky(row):
        if row["uid"] == "val-3-0":
            raise ValueError("unreadable clip")
        return fake_audio([])(row)

    fresh = SimpleNamespace(**{**vars(args), "output": tmp_path / "fresh", "per_label": 0})
    summary = cluster_speakers(fresh, embed=fake_embed, load=flaky)
    assert summary["embedding_failures"] == 1 and summary["failures"][0]["uid"] == "val-3-0"
    assert summary["embedded_utterances"] == 35

    def broken(row):
        raise ValueError("Incompatible codec/cache: checkpoint")

    with pytest.raises(ValueError, match="first 20 rows all failed"):
        cluster_speakers(SimpleNamespace(**{**vars(args), "output": tmp_path / "broken"}), embed=fake_embed, load=broken)
