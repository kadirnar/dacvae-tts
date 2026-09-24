"""Teacher-feature auxiliary losses (issue #10): stores, alignment with the latents, losses, training."""

import importlib.util
import json
import math
import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dacvae_tts.alignment import cosine_distance, masked_distance, teacher_terms, weighted_alignment
from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, flow_loss, sample
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
from dacvae_tts.training import Objective, load_model

HOP = 512  # the synthetic cache: 24 kHz, hop 512
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_teacher_features.py"
NANO = dict(
    latent_dim=4,
    width=32,
    heads=2,
    depth=2,
    text_depth=1,
    text_attention=1,
    patch_size=1,
    positions="rope",
    qk_norm=True,
    prediction="edm",
    text_layout="joined",
    duration="rule",
    cond_dropout=0.0,
)


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


def test_cosine_terms_on_toy_tensors():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    assert cosine_distance(a, a).abs().max() < 1e-6
    assert torch.allclose(cosine_distance(a, -a), torch.full((3,), 2.0))
    assert torch.allclose(cosine_distance(a, a.flip(-1)), torch.tensor([1.0, 1.0, 1.0]))
    prediction = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [5.0, 5.0]]])
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(masked_distance(prediction, target, mask), torch.tensor([1.0]))  # (0 + 2) / 2
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]])
    speaker = torch.tensor([[1.0, 0.0]])
    alignment, negentropy = weighted_alignment(embeddings, speaker, torch.full((1, 4), 0.25))
    assert torch.allclose(alignment, torch.tensor([(0 + 1 + 2 + (1 - 0.5**0.5)) / 4]))
    assert torch.allclose(negentropy, torch.tensor([-math.log(4)]))
    peaked, _ = weighted_alignment(embeddings, speaker, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert peaked.abs().max() < 1e-6 and weighted_alignment(embeddings, speaker, torch.eye(4)[:1])[1] == 0


def randomized(model):
    for block in model.blocks:
        torch.nn.init.normal_(block.ada[-1].weight, std=0.05)
    torch.nn.init.normal_(model.output[-1].weight, std=0.05)
    return model


def teacher_batch(cache):
    frames, speakers = build_stores(cache)
    data = LatentDataset(
        cache, pairing="within", layout="joined", teacher_features=frames, speaker_embeddings=speakers
    )
    return collate([data[(0, i)] for i in range(3)])


def test_teacher_terms_reach_the_generator_and_the_heads(cache):
    batch = teacher_batch(cache)
    model = randomized(FlowTTS(ModelConfig(**NANO, repa_layer=1, repa_dim=3, tla_layers="all", tla_dim=3)))
    objective = Objective(model, expansion=2, repa_weight=1.0, tla_weight=0.5).train()
    losses = objective(batch)
    assert losses["repa"].shape == losses["tla"].shape == losses["flow"].shape == (6,)
    assert (losses["repa"] > 0).all() and (losses["tla"] > 0).all()
    # The block-weight network starts at zero: uniform weights, maximum entropy.
    assert torch.allclose(losses["tla_entropy"], torch.full((6,), -math.log(2)))
    assert "teacher_idle" not in losses and "hidden" not in losses
    objective.auxiliary(losses).mean().backward()
    for module in (model.repa, model.tla.heads, model.tla.weights[-1], model.blocks[0], model.input):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.output[-1].weight.grad is None  # the teacher terms do not touch the velocity head

    model.zero_grad(set_to_none=True)
    repa_only = Objective(model, repa_weight=1.0).train()
    repa_only(batch)["repa"].mean().backward()
    after = model.blocks[1].parameters()  # the block after the REPA block gets nothing from it
    assert all(p.grad is None or not p.grad.any() for p in after)
    assert all(p.grad is not None for p in model.blocks[0].parameters())


def test_frame_selection_and_condition_dropout(cache):
    batch = teacher_batch(cache)
    assert batch["prompt_mask"].any(1).all()
    model = randomized(FlowTTS(ModelConfig(**NANO, repa_layer=1, repa_dim=3, tla_layers=[1, 2], tla_dim=3)))
    inputs = [batch[k] for k in ("prompt", "prompt_mask", "tokens", "segments")]
    time = torch.tensor([0.2, 0.5, 0.8])
    _, hidden = model(batch["latents"], time, *inputs[:2], batch["valid"], *inputs[2:], return_hidden=(1, 2))
    kept, dropped = torch.zeros(3, dtype=torch.bool), torch.tensor([True, False, False])
    every = teacher_terms(model, batch, hidden, time, kept)
    target = teacher_terms(model, batch, hidden, time, kept, repa_frames="target")
    blind = teacher_terms(model, batch, hidden, time, dropped)
    assert (every["repa"] != target["repa"]).all()  # prompt frames join the average under "all"
    # A condition-dropped example: its zeroed prompt is not aligned, and TLA-SA skips it.
    assert blind["repa"][0] == target["repa"][0] and torch.equal(blind["repa"][1:], every["repa"][1:])
    assert blind["tla"][0] == 0 and blind["tla_entropy"][0] == 0 and (blind["tla"][1:] > 0).all()


def test_repa_stop_step_and_idle_heads_keep_ddp_gradients(cache):
    batch = teacher_batch(cache)
    model = randomized(FlowTTS(ModelConfig(**NANO, repa_layer=1, repa_dim=3, tla_layers=[2], tla_dim=3)))
    objective = Objective(model, repa_weight=1.0, tla_weight=0.5).train()
    objective.repa_active = False  # the train loop's state from repa_stop_step on
    losses = objective(batch)
    assert "repa" not in losses and "tla" in losses and not losses["teacher_idle"].any()
    objective.auxiliary(losses).sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.repa.parameters())
    assert all(p.grad is not None and p.grad.any() for p in model.tla.heads.parameters())
    assert not Objective(model, repa_weight=1.0).eval()(batch).keys() & {"repa", "tla", "teacher_idle"}
    with pytest.raises(ValueError, match="teacher frames"):
        Objective(model, repa_weight=1.0).train()({k: v for k, v in batch.items() if k != "teacher"})


def one_item(prompt, target, text):
    item = dict(reference=torch.randn(prompt, 4), target=torch.randn(target, 4), text=text, reference_text="")
    return collate([{**item, "layout": "joined"}])


def test_options_off_leave_model_and_checkpoints_unchanged(tmp_path):
    torch.manual_seed(3)
    off = FlowTTS(ModelConfig(**NANO))
    torch.manual_seed(3)
    on = FlowTTS(ModelConfig(**NANO, repa_layer=2, repa_dim=3, tla_layers="all", tla_dim=3))
    assert not any(k.startswith(("repa", "tla")) for k in off.state_dict())
    shared = on.state_dict()
    for key, value in off.state_dict().items():
        assert torch.equal(value, shared[key]), key  # the heads are created after every other module
    missing = on.load_state_dict(randomized(off).state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith(("repa.", "tla.")) for k in missing)
    batch = one_item(3, 6, "Bir iki.")
    kwargs = {k: v for k, v in batch.items() if k != "latents"}
    time = torch.tensor([0.4])
    reference = off.eval()(batch["latents"], time, **kwargs)
    assert torch.equal(on.eval()(batch["latents"], time, **kwargs), reference)
    velocity, hidden = off(batch["latents"], time, **kwargs, return_hidden=(1, 2))
    assert torch.equal(velocity, reference) and set(hidden) == {1, 2} and hidden[2].shape == (1, 9, 32)
    torch.manual_seed(5)
    plain = flow_loss(off.train(), batch, return_details=True)
    torch.manual_seed(5)
    probed = flow_loss(off, batch, return_details=True, hidden_layers=(1,))
    assert torch.equal(plain["flow"], probed["flow"])
    assert "hidden" not in plain and set(probed["hidden"]) == {1}

    # A checkpoint written before these options existed: its config lacks every new key.
    old = Config(ModelConfig(**NANO), TrainConfig(pairing="within")).to_dict()
    for key in ("repa_layer", "repa_dim", "tla_layers", "tla_dim", "tla_hidden"):
        old["model"].pop(key)
    for key in ("teacher_features", "repa_weight", "repa_stop_step", "repa_frames", "speaker_embeddings"):
        old["train"].pop(key)
    old["train"].pop("tla_weight"), old["train"].pop("tla_entropy")
    torch.save({"config": old, "model": off.state_dict(), "ema": off.state_dict()}, tmp_path / "old.pt")
    loaded, _ = load_model(tmp_path / "old.pt")  # strict load_state_dict
    assert loaded.repa is None and loaded.tla is None


def test_sampler_never_calls_the_teacher_heads():
    model = FlowTTS(ModelConfig(**NANO, repa_layer=2, repa_dim=3, tla_layers="all", tla_dim=3)).eval()

    def forbidden(*_):
        raise AssertionError("teacher head called at inference")

    model.repa.register_forward_hook(forbidden)
    for module in model.tla.modules():
        module.register_forward_hook(forbidden)
    batch = one_item(3, 5, "Merhaba.")
    prompt = batch["latents"] * batch["prompt_mask"][..., None]
    out = sample(
        model, prompt, batch["prompt_mask"], batch["valid"], batch["tokens"], batch["segments"], steps=2
    )
    assert torch.isfinite(out).all()


def test_configuration_guards():
    assert ModelConfig(depth=4, tla_layers="all", tla_dim=8).tla_layers == (1, 2, 3, 4)
    assert ModelConfig(depth=4, tla_layers=[3, 1, 3], tla_dim=8).tla_layers == (1, 3)
    cfg = Config(
        ModelConfig(repa_layer=6, repa_dim=16, tla_layers=[2, 4], tla_dim=8, ctc_layer=4),
        TrainConfig(teacher_features="t", repa_weight=1.0, speaker_embeddings="s", tla_weight=0.5),
    )
    assert Config.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg  # JSON lists vs tuples
    assert Config.from_dict(yaml.safe_load(yaml.safe_dump(json.loads(json.dumps(cfg.to_dict()))))) == cfg
    bad = [
        lambda: ModelConfig(repa_layer=4, repa_dim=16, ctc_layer=4),  # A-DMA: never on the CTC block
        lambda: ModelConfig(repa_layer=4),  # no teacher width
        lambda: ModelConfig(repa_layer=9, repa_dim=16),  # beyond the depth
        lambda: ModelConfig(tla_layers=[0, 2], tla_dim=8),
        lambda: ModelConfig(tla_layers="some", tla_dim=8),
        lambda: ModelConfig(tla_layers="all"),  # no embedding width
        lambda: TrainConfig(repa_weight=1.0),  # no store
        lambda: TrainConfig(repa_frames="prompt"),
        lambda: TrainConfig(tla_weight=-1.0, speaker_embeddings="s"),
        lambda: Config(ModelConfig(), TrainConfig(teacher_features="t", repa_weight=1.0)),  # no projector
    ]
    for build in bad:
        with pytest.raises(ValueError):
            build()


def train_args(config, cache, output):
    return types.SimpleNamespace(
        config=str(config),
        cache=str(cache),
        output=str(output),
        resume=None,
        init_from=None,
        device="cpu",
        steps=None,
        batch_size=None,
        accumulation=None,
        workers=None,
        precision=None,
        learning_rate=None,
        optimizer=None,
        worker_threads=None,
        prefetch_factor=None,
        loader_start_method=None,
        cuda_prefetch=None,
        frame_budget=0,
        compile=None,
        no_validation=False,
        stop_after=None,
    )


def test_training_logs_teacher_terms_and_checkpoints_the_heads(cache, tmp_path):
    from dacvae_tts import training

    build_stores(cache)
    config = tmp_path / "teacher.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "latent_dim": 4,
                    "width": 16,
                    "depth": 2,
                    "heads": 2,
                    "text_depth": 1,
                    "repa_layer": 2,
                    "repa_dim": 3,
                    "tla_layers": "all",
                    "tla_dim": 3,
                    "tla_hidden": 8,
                },
                "train": {
                    "steps": 4,
                    "warmup": 1,
                    "batch_size": 2,
                    "accumulation": 2,
                    "workers": 0,
                    "precision": "fp32",
                    "log_every": 1,
                    "checkpoint_every": 2,
                    "validate_every": 2,
                    "teacher_features": "teacher/frames",
                    "repa_weight": 1.0,
                    "repa_stop_step": 3,
                    "speaker_embeddings": "teacher/speakers",
                    "tla_weight": 0.5,
                },
            }
        )
    )
    training.train(train_args(config, cache, tmp_path / "run"))
    records = [json.loads(line) for line in (tmp_path / "run" / "train.jsonl").read_text().splitlines()]
    steps = [r for r in records if "flow" in r]
    assert [r["repa"] > 0 for r in steps] == [True, True, True, False]  # HASTE: the first 3 updates only
    assert all(r["tla"] > 0 and r["tla_entropy"] < 0 for r in steps)
    assert any("validation_loss" in r for r in records)
    model, saved = load_model(tmp_path / "run" / "last.pt")
    assert any(k.startswith("repa.") for k in saved["ema"])
    assert any(k.startswith("tla.") for k in saved["model"])
    assert Config.from_dict(saved["config"]) == Config.load(config)  # resume compares these
    assert model.cfg.tla_layers == (1, 2)


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


@pytest.mark.parametrize("name", ["repa", "tla", "repa_tla"])
def test_example_configs_differ_from_w512_only_in_teacher_options(name):
    root = Path(__file__).resolve().parents[1] / "configs"
    base = Config.load(root / "nano_tr_w512.yaml")
    cfg = Config.load(root / "experiments" / f"tr_w512_{name}.yaml")
    teacher = {"repa_layer", "repa_dim", "tla_layers", "tla_dim", "tla_hidden"}
    teacher |= {"teacher_features", "repa_weight", "repa_stop_step", "repa_frames"}
    teacher |= {"speaker_embeddings", "tla_weight", "tla_entropy"}
    for part in ("model", "train"):
        changed = {k for k, v in vars(getattr(cfg, part)).items() if getattr(getattr(base, part), k) != v}
        assert changed and changed <= teacher, changed
    assert (cfg.train.repa_weight > 0) == ("repa" in name) and (cfg.train.tla_weight > 0) == ("tla" in name)
