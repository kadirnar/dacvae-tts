"""Frozen speaker-embedding condition (model.speaker_condition_dim): model, data, config, optimizer, inference."""

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, sample, text_only_rows
from dacvae_tts.optim import muon_parts
from dacvae_tts.teacher import TeacherStore
from dacvae_tts.text import tokenize
from dacvae_tts.training import speaker_metadata

SMALL = dict(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined", duration="rule",
             positions="rope", prediction="edm", adaln_rank=4)


def inputs(batch=2, frames=9, prompt=4):
    tokens, segments = tokenize("Önceki cümle.", "Yeni bir cümle.", "turkish-v1", "joined")
    x = torch.randn(batch, frames, 4)
    mask = torch.arange(frames)[None].expand(batch, -1) < prompt
    return dict(x=x, time=torch.full((batch,), 0.4), prompt=x * mask[..., None], prompt_mask=mask,
                valid=torch.ones(batch, frames, dtype=torch.bool), tokens=tokens[None].repeat(batch, 1),
                segments=segments[None].repeat(batch, 1))


def test_zero_init_same_seed_equals_the_baseline():
    torch.manual_seed(0)
    base = FlowTTS(ModelConfig(**SMALL))
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**SMALL, speaker_condition_dim=3))
    shared = {k: v for k, v in model.state_dict().items() if not k.startswith("speaker_condition")}
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items())  # no extra random draws
    assert set(model.state_dict()) - set(base.state_dict()) == {"speaker_condition.weight"}
    batch = inputs()
    with torch.no_grad():
        expected = base(**batch)
        for speaker in (torch.zeros(2, 3), torch.randn(2, 3)):
            assert torch.equal(model(**batch, speaker=speaker), expected)
    assert muon_parts("speaker_condition.weight", model.speaker_condition.weight) == 0  # AdamW, like `input`


def test_the_embedding_conditions_prompted_rows_and_every_null_path_drops_it():
    torch.manual_seed(1)
    model = FlowTTS(ModelConfig(**SMALL, speaker_condition_dim=3)).eval()
    with torch.no_grad():  # a trained-looking model: the zero-init heads would hide every condition
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    batch = inputs()
    a, b = torch.randn(2, 3), torch.randn(2, 3)
    with torch.no_grad():
        assert not torch.allclose(model(**batch, speaker=a), model(**batch, speaker=b))
        drop = torch.ones(2, dtype=torch.bool)  # CFG null branch / condition dropout
        assert torch.equal(model(**batch, speaker=a, drop=drop), model(**batch, speaker=b, drop=drop))
        voice = lambda speaker, mask: model.reference_summary(batch["prompt"], mask, speaker)  # noqa: E731
        empty = torch.zeros_like(batch["prompt_mask"])  # prompt dropout: no voice at all
        assert torch.equal(voice(a, empty), voice(b, empty)) and torch.equal(voice(a, empty), voice(0 * a, empty))
        scale = voice(a * 5, batch["prompt_mask"])  # L2-normalized: the scale of the embedding does not matter
        assert torch.allclose(scale, voice(a, batch["prompt_mask"]), atol=1e-6)
        rows = text_only_rows(model, batch["valid"], batch["prompt_mask"], batch["tokens"], batch["segments"])
        assert torch.equal(rows["cached"][2], torch.zeros_like(rows["cached"][2]))  # prompt-free branch: no voice
        common = dict(prompt=batch["prompt"], prompt_mask=batch["prompt_mask"], valid=batch["valid"],
                      tokens=batch["tokens"], segments=batch["segments"], steps=3, guidance=2.0, seed=0)
        assert not torch.allclose(sample(model, **common, speaker=a), sample(model, **common, speaker=b))
    with pytest.raises(ValueError, match="speaker embedding"):
        model(**batch)
    with pytest.raises(ValueError, match="speaker embedding"):
        model(**batch, speaker=torch.zeros(2, 5))


def speaker_labels(cache):
    with sqlite3.connect(cache / "index.sqlite") as db:
        return dict(db.execute("SELECT uid, speaker FROM samples WHERE split='train'"))


def test_items_are_conditioned_on_the_prompt_speaker(cache):
    from test_teacher import build_stores

    _, store = build_stores(cache, ("train", "val"))
    vectors, labels = TeacherStore(store, "speaker"), speaker_labels(cache)
    within = LatentDataset(cache, "train", 42, "within", "joined", speaker_condition=store)
    for index in range(len(within)):
        item = within[(0, index)]
        vector = item["speaker_condition"]
        matches = [u for u in labels if torch.equal(vectors.speaker(u), vector)]
        assert len(matches) == 1 and matches[0] != item["uid"] and labels[matches[0]] == labels[item["uid"]]
    same = LatentDataset(cache, "train", 42, "within", "joined", speaker_condition=store,
                         speaker_condition_source="same")
    assert torch.equal(same[(0, 0)]["speaker_condition"], vectors.speaker(same[(0, 0)]["uid"]))
    strict = LatentDataset(cache, "train", 42, "within", "joined", speaker_condition=store,
                           speaker_condition_min_cosine=1.0)
    item = strict[(0, 0)]  # no other utterance is that close: the item's own embedding
    assert torch.equal(item["speaker_condition"], vectors.speaker(item["uid"]))
    silent = LatentDataset(cache, "train", 42, "within", "joined", prompt_dropout=1.0, speaker_condition=store)
    assert torch.equal(silent[(0, 0)]["speaker_condition"], torch.zeros(3))
    cross = LatentDataset(cache, "train", 42, "cross", "joined", speaker_condition=store)
    item = cross[(0, 1)]
    assert torch.allclose(item["speaker_condition"], F.normalize(vectors.speaker(item["reference_uid"]), dim=0))
    prompts = LatentDataset(cache, "train", 42, "within", "joined", speaker_condition=store, cross_prompt_prob=1.0)
    item = prompts[(0, 2)]
    refs = torch.stack([F.normalize(vectors.speaker(u), dim=0) for u in item["reference_uid"].split("|")])
    assert torch.allclose(item["speaker_condition"], F.normalize(refs.mean(0), dim=0))
    batch = collate([within[(0, i)] for i in range(3)])
    assert batch["speaker_condition"].shape == (3, 3)
    plain = LatentDataset(cache, "train", 42, "within", "joined")[(0, 0)]
    assert "speaker_condition" not in plain and "speaker_condition" not in collate([plain])
    with pytest.raises(ValueError, match="speaker_condition"):
        collate([within[(0, 0)], plain])
    assert speaker_metadata(within)["speaker_condition"]["dim"] == 3 and speaker_metadata(
        LatentDataset(cache, "train")) == {}


def test_speaker_condition_options_are_validated():
    model = ModelConfig(**SMALL, speaker_condition_dim=192)
    train = TrainConfig(pairing="within", speaker_condition="teacher/campplus")
    assert Config(model, train).model.speaker_condition_dim == 192
    for bad in ((model, TrainConfig(pairing="within")), (ModelConfig(**SMALL), train)):
        with pytest.raises(ValueError, match="go together"):
            Config(*bad)
    with pytest.raises(ValueError, match="another store"):
        TrainConfig(speaker_condition="teacher/ecapa", speaker_embeddings="teacher/ecapa", tla_weight=0.5)
    with pytest.raises(ValueError):
        TrainConfig(speaker_condition_source="target")
    with pytest.raises(ValueError):
        ModelConfig(**SMALL, speaker_condition_dim=-1)


class FakeEmbedder:
    sample_rate = 16000
    calls = 0

    def __init__(self, model=None, device="cpu"):
        pass

    def __call__(self, waveform):
        FakeEmbedder.calls += 1
        return torch.tensor([float(np.mean(waveform)), float(np.std(waveform)), len(waveform) / 1e4])


def test_inference_embeds_each_prompt_once_and_batches_carry_it():
    from dacvae_tts.inference import Synthesizer, VoiceReference

    tts = Synthesizer.__new__(Synthesizer)
    tts.model = FlowTTS(ModelConfig(**SMALL, speaker_condition_dim=3))
    tts.device, tts.mean, tts.std = torch.device("cpu"), torch.zeros(4), torch.ones(4)
    tts.codec = SimpleNamespace(sample_rate=2500, hop_length=100, latent_dim=4,
                                decode=lambda z: z[:, 0].repeat_interleave(100))
    tts.text_version, tts.duration_model = "turkish-v1", None
    tts.speaker_record = {"embedder": "test_speaker_condition:FakeEmbedder", "model": None, "dim": 3}
    voice = VoiceReference(torch.randn(20, 4), "Önceki cümle.", "provided", {})
    FakeEmbedder.calls = 0
    first = tts.speaker_embedding(voice)
    assert first.shape == (3,) and torch.equal(tts.speaker_embedding(voice), first) and FakeEmbedder.calls == 1
    batch = tts.make_batch(voice.latents, voice.transcript, "Yeni bir cümle.", speaker=first)
    assert torch.equal(batch["speaker"], first[None])
    plain = Synthesizer.__new__(Synthesizer)
    plain.model = FlowTTS(ModelConfig(**SMALL))
    assert plain.speaker_embedding(voice) is None
