"""Prompt tempo perturbation: tempo-variant store (dacvae_tts.tempo) and its LatentDataset integration, no model."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dacvae_tts.config import TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.teacher import CacheAudio
from dacvae_tts.tempo import TempoStore, build_tempo_variants, merge_tempo_parts, tempo_key

HOP, RATE = 512, 24000  # the conftest cache: 7/9/11-frame rows, 4 speakers x 3 utterances per split
TEMPOS = (800, 1000, 1250)


def fake_decode(latents):
    return latents[:, 0].repeat_interleave(HOP)


def fake_encode(waveform):
    """Per-hop means as 4 channels: deterministic, and a function of the stretched audio's length."""
    frames = math.ceil(len(waveform) / HOP)
    padded = np.pad(waveform, (0, frames * HOP - len(waveform)))
    means = padded.reshape(frames, HOP).mean(1)
    return np.stack([means, means * 0.5, -means, means + 1.0], 1)


def build(cache, output=None, tempos=TEMPOS, **kwargs):
    output = output or cache / "tempo" / "wsola-v1"
    build_tempo_variants(cache, output, fake_encode, CacheAudio(cache, "decode", fake_decode), tempos, **kwargs)
    return output


def within(cache, store=None, **options):
    options.setdefault("prompt_fraction", (0.2, 0.6))
    return LatentDataset(cache, "train", 42, pairing="within", layout="joined", tempo_variants=store, **options)


def test_store_build_resume_merge_and_checks(cache, tmp_path):
    store = build(cache)
    meta = json.loads((store / "metadata.json").read_text())
    assert meta["kind"] == "tempo" and meta["tempos"] == list(TEMPOS) and meta["complete"] and meta["merged"]
    assert meta["algorithm"] == {"name": "wsola", "frame_ms": 40.0, "tolerance_ms": 15.0}
    reader = TempoStore(store)
    for tempo in TEMPOS:  # frames of every record = ceil(round(samples / t) / hop)
        lengths = reader.lengths(["train-0-0", "train-3-2"], tempo)
        assert lengths.tolist() == [math.ceil(round(f * HOP * 1000 / tempo) / HOP) for f in (7, 11)]
    record = reader.latents("train-1-1", 1000)  # float16 on disk: channel 3 = channel 0 + 1 up to rounding
    assert torch.allclose(record[:, 3], record[:, 0] + 1, atol=2e-3)
    assert build(cache) == store  # complete: returned as is
    # Two partitions merged give the same records as one build.
    parts = tmp_path / "parts"
    for index in range(2):
        build(cache, parts, shard_index=index, num_shards=2)
    merged = merge_tempo_parts(parts, cache)
    assert merged["rows"] == 12 and TempoStore(parts).meta["partition"] is None
    for uid in ("train-0-0", "train-2-1"):
        assert torch.equal(TempoStore(parts).latents(uid, 800), reader.latents(uid, 800))
    reader.check(cache / "index.sqlite", "train", TEMPOS)
    with pytest.raises(ValueError, match="tempos"):
        reader.check(cache / "index.sqlite", "train", (900,))
    with pytest.raises(ValueError, match="rows have no"):
        reader.check(cache / "index.sqlite", "val", (1000,))  # built for train only
    with pytest.raises(ValueError, match="another cache"):
        reader.check(cache / "index.sqlite", "train", TEMPOS, {"index_sha256": "different"} | {"checkpoint": "x"})


def test_tempo_off_keeps_items_and_costs(cache):
    store = build(cache)
    plain, off = within(cache), within(cache, store, tempo_prompt_prob=0.0)
    assert np.array_equal(plain.costs, off.costs)
    for epoch in range(2):
        for index in range(len(plain)):
            a, b = plain[(epoch, index)], off[(epoch, index)]
            assert a.keys() == b.keys() and all(
                torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def test_within_prompts_are_the_stretched_prefix_and_keep_the_target(cache):
    store = build(cache)
    plain = within(cache)
    stretched = within(cache, store, tempo_prompt_prob=1.0)
    reader, seen = TempoStore(store), set()
    for epoch in range(3):
        for index in range(len(plain)):
            base, item = plain[(epoch, index)], stretched[(epoch, index)]
            assert torch.equal(base["target"], item["target"]) and base["text"] == item["text"]
            tempo = round(item["prompt_tempo"] * 1000)
            cut = len(base["reference"])
            variant = (reader.latents(item["uid"], tempo) - plain.mean) / plain.std
            assert torch.equal(item["reference"], variant[: min(max(round(cut * 1000 / tempo), 1), len(variant))])
            seen.add(tempo)
    assert seen == set(TEMPOS)
    only = within(cache, store, tempo_prompt_prob=1.0, tempo_prompt_factors=(1.25,))
    assert {only[(0, i)]["prompt_tempo"] for i in range(len(only))} == {1.25}
    none = within(cache, store, tempo_prompt_prob=1.0, prompt_dropout=1.0)
    assert all(len(none[(0, i)]["reference"]) == 0 and "prompt_tempo" not in none[(0, i)] for i in range(len(none)))


@pytest.mark.parametrize("options", [
    dict(),
    dict(prompt_cut="quiet", tail_silence_prob=0.5),
    dict(long_prompt_prob=0.5, prompt_fraction_long_max=0.85),
    dict(cross_prompt_prob=0.5, cross_prompt_max_seconds=0.5),
])
def test_stretched_items_stay_within_epoch_and_static_costs(cache, options):
    if "prompt_cut" in options:
        from test_pairs import write_silence

        write_silence(cache)
    store = build(cache)
    data = within(cache, store, tempo_prompt_prob=0.7, **options)
    for epoch in range(4):
        costs = data.epoch_costs(epoch)
        assert np.all(costs <= data.costs)
        for index in range(len(data)):
            item = data[(epoch, index)]
            assert len(item["reference"]) + len(item["target"]) <= costs[index]


def test_cross_prompts_stretch_every_reference(cache):
    store = build(cache)
    data = within(cache, store, tempo_prompt_prob=1.0, cross_prompt_prob=1.0, cross_prompt_max_seconds=1.0)
    reader = TempoStore(store)
    for index in range(len(data)):
        item = data[(0, index)]
        tempo = round(item["prompt_tempo"] * 1000)
        refs = item["reference_uid"].split("|")
        expected = torch.cat([(reader.latents(uid, tempo) - data.mean) / data.std for uid in refs])
        assert torch.equal(item["reference"], expected) and len(expected) <= data.cross_prompt_frames
    cross_only = within(cache, store, tempo_prompt_prob=1.0, tempo_prompt_pairs="cross", cross_prompt_prob=0.5)
    items = [cross_only[(0, i)] for i in range(len(cross_only))]
    assert all(("prompt_tempo" in item) == ("|" in item["reference_uid"] or item["reference_uid"] != item["uid"])
               for item in items)


def test_stretched_prompts_have_no_teacher_frames(cache):
    from test_teacher import build_stores

    frames, _ = build_stores(cache)
    store = build(cache)
    data = within(cache, store, tempo_prompt_prob=1.0, teacher_features=frames)
    items = [data[(0, i)] for i in range(4)]
    batch = collate(items)
    for row, item in enumerate(items):
        prompt = len(item["reference"])
        assert not batch["teacher_valid"][row, :prompt].any()
        assert batch["teacher_valid"][row, prompt : prompt + len(item["target"])].all()
        assert (batch["teacher"][row, :prompt] == 0).all()


def test_tempo_options_are_validated():
    ok = dict(pairing="within", tempo_variants="tempo/wsola-v1")
    assert TrainConfig(tempo_prompt_prob=0.3, **ok).tempo_prompt_factors == ()
    assert TrainConfig(tempo_prompt_prob=0.3, tempo_prompt_factors=[0.8, 1.25], **ok).tempo_prompt_factors == (0.8, 1.25)
    for bad in (dict(tempo_prompt_prob=0.3, pairing="within"), dict(tempo_prompt_prob=0.3, tempo_variants="x"),
                dict(tempo_prompt_prob=1.5, **ok), dict(tempo_prompt_prob=0.3, tempo_prompt_factors=[3.0], **ok),
                dict(tempo_prompt_prob=0.3, tempo_prompt_pairs="cross", **ok)):
        with pytest.raises(ValueError):
            TrainConfig(**bad)
    assert tempo_key(1.25) == 1250 and tempo_key(800) == 800
    with pytest.raises(ValueError):
        tempo_key(0.3)


def test_build_script_builds_and_merges_with_a_codec(cache, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_tempo_variants.py"
    spec = importlib.util.spec_from_file_location("build_tempo_variants_under_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    class FakeCodec:
        def encode(self, audio):
            return torch.from_numpy(fake_encode(audio.numpy()))

        def decode(self, latents):
            return fake_decode(latents)

    monkeypatch.setattr(script, "load_codec", lambda cache, checkpoint, device: FakeCodec())
    output = cache / "tempo" / "cli"
    for index in range(2):
        script.main(["build", "--cache", str(cache), "--output", str(output), "--tempos", "0.8", "1.0",
                     "--device", "cpu", "--shard-index", str(index), "--num-shards", "2", "--quiet"])
    result = script.main(["merge", "--output", str(output), "--cache", str(cache)])
    assert result["tempos"] == [800, 1000] and result["audio_sources"] == {"original": 0, "decoded": 12}
