"""Teacher-feature stores (issue #10): extraction, alignment with the latents, dataset and collate."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.teacher import (
    CacheAudio,
    TeacherStore,
    apply_pca,
    cache_rows,
    extract_frames,
    extract_speakers,
    fit_pca,
    merge_parts,
    pool_frames,
)

HOP = 512  # the synthetic cache: 24 kHz, hop 512

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_teacher_features.py"


def fake_decode(latents):
    """Piecewise-constant 'waveform': channel 0 of each latent frame held for one hop."""
    return latents[:, 0].repeat_interleave(HOP)


class FakeTeacher:
    """Two frames per latent frame and one frame short at the end, like a HuBERT window."""

    sample_rate = 24000
    frame_rate = 2 * 24000 / HOP
    layer = 12

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def __call__(self, waveform):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash")
        hop = HOP // 2
        count = len(waveform) // hop - 1
        value = torch.from_numpy(waveform[: count * hop].reshape(count, hop).mean(1))
        return torch.stack([value, -value, (torch.arange(count) % 2).float()], 1)


class FakeEmbedder:
    sample_rate = 24000

    def __call__(self, waveform):
        return torch.tensor([waveform.mean(), waveform.std(), len(waveform) / 1e4])


def raw_latents(cache):
    with sqlite3.connect(cache / "index.sqlite") as db:
        rows = db.execute("SELECT uid, shard, offset, frames FROM samples").fetchall()
    return {
        uid: torch.from_numpy(
            np.memmap(shard, dtype="<f2", mode="r").reshape(-1, 4)[o : o + f].astype(np.float32)
        )
        for uid, shard, o, f in rows
    }


def build_stores(cache, splits=("train",)):
    frames, speakers = cache / "teacher" / "frames", cache / "teacher" / "speakers"
    extract_frames(cache, frames, FakeTeacher(), CacheAudio(cache, "decode", fake_decode), splits)
    extract_speakers(cache, speakers, FakeEmbedder(), CacheAudio(cache, "auto", fake_decode), splits)
    return frames, speakers


def test_pool_frames_pairs_pads_and_rejects_mismatches():
    features = torch.arange(9.0)[:, None].repeat(1, 2)  # 2 * 5 - 1 frames at twice the latent rate
    pooled = pool_frames(features, 2.0, 5)
    assert pooled[:, 0].tolist() == [0.5, 2.5, 4.5, 6.5, 8.0]  # the missing tail repeats frame 8
    assert pool_frames(torch.randn(20, 3), 50 / 23, 9).shape == (9, 3)  # non-integer ratio
    with pytest.raises(ValueError):
        pool_frames(torch.randn(30, 2), 2.0, 5)


def test_pca_recovers_the_dominant_directions():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(6, 2))[0]
    x = (torch.randn(500, 2) * torch.tensor([5.0, 2.0])) @ basis.T + 3 + 0.01 * torch.randn(500, 6)
    x = x.double()
    pca = fit_pca(x.sum(0), x.T @ x, len(x), 2)
    assert pca["explained"].sum() > 0.999
    assert torch.allclose(pca["components"].T @ pca["components"], torch.eye(2), atol=1e-5)
    projected = apply_pca(x.float(), pca)
    assert projected.shape == (500, 2) and projected.mean(0).abs().max() < 1e-3  # centred
    again = fit_pca(x.sum(0), x.T @ x, len(x), 2)
    assert torch.equal(pca["components"], again["components"])  # deterministic signs


def test_extraction_aligns_frames_with_latents_and_resumes(cache):
    latents = raw_latents(cache)
    output = cache / "teacher" / "frames"
    with pytest.raises(RuntimeError):
        extract_frames(cache, output, FakeTeacher(fail_after=5), CacheAudio(cache, "decode", fake_decode))
    assert not (output / "metadata.json").exists()
    teacher = FakeTeacher()
    meta = extract_frames(cache, output, teacher, CacheAudio(cache, "decode", fake_decode))
    assert teacher.calls == 12 - 5  # resumed: the five rows written before the crash are skipped
    assert meta["rows"] == 12 and meta["dim"] == 3 and meta["pooling"] == "mean-2" and meta["merged"]
    assert meta["frame_rate"] == 24000 / HOP and meta["splits"] == ["train"]
    store = TeacherStore(output, "frames")
    for uid, z in latents.items():
        if uid.startswith("train"):
            features = store.frames(uid, len(z)).float()
            assert torch.equal(features[:, 0], z[:, 0]) and torch.equal(features[:, 1], -z[:, 0])
            assert features[:-1, 2].eq(0.5).all() and features[-1, 2] == 0  # padded tail frame
    with pytest.raises(ValueError):
        store.frames("train-0-0", 3)  # a frame count different from the latents'
    with pytest.raises(ValueError):
        TeacherStore(output, "speaker")
    with pytest.raises(ValueError, match="not available"):
        CacheAudio(cache, "original", fake_decode)(next(cache_rows(cache)))  # its audio.wav does not exist


def test_sharded_extraction_merges_and_checks_coverage(cache):
    output = cache / "teacher" / "speakers"
    audio = CacheAudio(cache, "decode", fake_decode)
    extract_speakers(cache, output, FakeEmbedder(), audio, shard_index=0, num_shards=2)
    with pytest.raises(ValueError):
        merge_parts(output, cache)  # partition 1 is missing
    with pytest.raises(ValueError):
        TeacherStore(output / "part-000-of-002", "speaker")  # a partition alone is not a store
    extract_speakers(cache, output, FakeEmbedder(), audio, shard_index=1, num_shards=2)
    meta = merge_parts(output, cache)
    assert meta["rows"] == 12 and meta["missing_rows"] == 0 and meta["audio_sources"]["decoded"] == 12
    store = TeacherStore(output, "speaker")
    z = raw_latents(cache)["train-2-1"]
    wave = fake_decode(z).numpy()
    assert torch.allclose(
        store.speaker("train-2-1"), torch.tensor([wave.mean(), wave.std(), len(wave) / 1e4])
    )
    with pytest.raises(ValueError):
        merge_parts(output, cache)  # already merged

    frames = cache / "teacher" / "frames"
    for index in range(2):
        extract_frames(cache, frames, FakeTeacher(), audio, shard_index=index, num_shards=2)
    with sqlite3.connect(frames / "part-001-of-002" / "index.sqlite") as db:
        db.execute("DELETE FROM teacher_features WHERE uid='train-3-2'")
    with pytest.raises(ValueError, match="no entry"):
        merge_parts(frames, cache)
    merged = merge_parts(frames, cache, allow_missing=True)
    assert merged["missing_rows"] == 1
    with pytest.raises(ValueError, match="no frames targets"):
        LatentDataset(cache, teacher_features=frames)  # the dataset refuses an incomplete store


@pytest.mark.parametrize("pairing", ["within", "cross"])
def test_dataset_slices_teacher_targets_like_latents(cache, pairing):
    frames, speakers = build_stores(cache)
    layout = "joined" if pairing == "within" else "segments"
    plain = LatentDataset(cache, pairing=pairing, layout=layout)
    data = LatentDataset(
        cache, pairing=pairing, layout=layout, teacher_features=frames, speaker_embeddings=speakers
    )
    raw, store = raw_latents(cache), TeacherStore(speakers, "speaker")
    items = [data[(0, i)] for i in range(4)]
    for index, item in enumerate(items):
        base = plain[(0, index)]
        assert set(item) - set(base) == {"teacher_reference", "teacher_target", "speaker_embedding"}
        assert all(torch.equal(item[k], base[k]) for k in ("reference", "target"))
        assert len(item["teacher_reference"]) == len(item["reference"])
        assert len(item["teacher_target"]) == len(item["target"])
        whole = torch.cat([item["teacher_reference"], item["teacher_target"]]).float()
        if pairing == "within":
            assert torch.equal(whole[:, 0], raw[item["uid"]][:, 0])
        else:
            assert torch.equal(item["teacher_reference"][:, 0].float(), raw[item["reference_uid"]][:, 0])
            assert torch.equal(item["teacher_target"][:, 0].float(), raw[item["uid"]][:, 0])
        assert torch.equal(item["speaker_embedding"], store.speaker(item["uid"]))
    batch = collate(items)
    assert set(batch) - set(collate([plain[(0, i)] for i in range(4)])) == {"teacher", "speaker_embedding"}
    assert batch["teacher"].shape[:2] == batch["latents"].shape[:2]
    assert batch["speaker_embedding"].shape == (4, 3)
    restored = batch["latents"] * data.std + data.mean
    valid = batch["valid"]
    assert torch.allclose(batch["teacher"][..., 0].float()[valid], restored[..., 0][valid], atol=1e-5)
    assert not batch["teacher"][~valid].any()  # padding stays zero
    if pairing == "within":
        dropped = LatentDataset(
            cache, pairing="within", layout="joined", prompt_dropout=1.0, teacher_features=frames
        )[(0, 0)]
        assert len(dropped["teacher_reference"]) == 0
        assert len(dropped["teacher_target"]) == len(dropped["target"])
    with pytest.raises(ValueError):
        collate([items[0], plain[(0, 1)]])  # every item or none carries teacher targets


def test_missing_split_or_store_fails_at_construction(cache):
    frames, speakers = build_stores(cache)
    with pytest.raises(ValueError, match="no frames targets"):
        LatentDataset(cache, "val", teacher_features=frames)  # only train was extracted
    with pytest.raises(ValueError, match="No teacher store"):
        LatentDataset(cache, teacher_features=cache / "teacher" / "absent")
    meta = json.loads((speakers / "metadata.json").read_text())
    (speakers / "metadata.json").write_text(json.dumps({**meta, "complete": False}))
    with pytest.raises(ValueError, match="Incomplete"):
        LatentDataset(cache, speaker_embeddings=speakers)


def load_script():
    spec = importlib.util.spec_from_file_location("extract_teacher_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extraction_script_runs_with_fake_models(cache, monkeypatch, capsys):
    script = load_script()
    loads = []
    monkeypatch.setattr(script, "load_frame_teacher", lambda args: FakeTeacher())
    monkeypatch.setattr(script, "load_decoder", lambda *a: loads.append(a) or fake_decode)
    monkeypatch.setattr(script, "load_embedder", lambda args: FakeEmbedder())

    def run(command, output, *flags):
        return script.main([command, "--cache", str(cache), "--output", str(output), "--quiet", *flags])

    output = cache / "teacher" / "pca"
    meta = run("frames", output, "--pca-dim", "2", "--pca-rows", "6")
    assert meta["dim"] == 2 and meta["pca"]["dim"] == 2 and (output / "pca.pt").exists()
    assert len(loads) == 1  # DACVAE is loaded once, lazily, for rows without an original file
    assert TeacherStore(output, "frames").frames("train-1-1", 9).shape == (9, 2)
    with pytest.raises(ValueError):
        run("frames", output, "--pca-dim", "3")  # the store already holds a 2-d PCA
    with pytest.raises(ValueError):
        run("frames", cache / "teacher" / "x", "--pca-dim", "2", "--num-shards", "2")  # fit-pca first
    speakers = cache / "teacher" / "spk"
    for index in ("0", "1"):
        run("speakers", speakers, "--shard-index", index, "--num-shards", "2")
    merged = script.main(["merge", "--output", str(speakers), "--cache", str(cache)])
    assert merged["rows"] == 12 and merged["embedder"] == "speechbrain" and merged["kind"] == "speaker"
    assert '"kind": "speaker"' in capsys.readouterr().out
