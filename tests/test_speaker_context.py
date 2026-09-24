"""Multi-clip speaker context (model.speaker_context: vector): model, data, config, warm start, optimizer, inference."""

import sqlite3
from types import SimpleNamespace

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, sample
from dacvae_tts.optim import muon_parts
from dacvae_tts.text import tokenize
from dacvae_tts.training import warm_start, warm_start_config

SMALL = dict(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined", duration="rule",
             positions="rope", prediction="edm", adaln_rank=4)
CONTEXT = dict(speaker_context="vector", speaker_context_width=8, speaker_context_layers=2, speaker_context_heads=2,
               speaker_context_patch=2)


def inputs(batch=2, frames=9, prompt=4):
    tokens, segments = tokenize("Önceki cümle.", "Yeni bir cümle.", "turkish-v1", "joined")
    x = torch.randn(batch, frames, 4)
    mask = torch.arange(frames)[None].expand(batch, -1) < prompt
    return dict(x=x, time=torch.full((batch,), 0.4), prompt=x * mask[..., None], prompt_mask=mask,
                valid=torch.ones(batch, frames, dtype=torch.bool), tokens=tokens[None].repeat(batch, 1),
                segments=segments[None].repeat(batch, 1))


def trained(model):
    with torch.no_grad():  # a trained-looking model: its zero-init heads would hide every condition
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    return model.eval()


def test_same_seed_and_zero_init_equal_the_baseline():
    torch.manual_seed(0)
    base = FlowTTS(ModelConfig(**SMALL))
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**SMALL, **CONTEXT))
    extra = set(model.state_dict()) - set(base.state_dict())
    assert extra and all(key.startswith("speaker_context.") for key in extra)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())  # created last
    batch, context = inputs(), torch.randn(2, 10, 4)
    with torch.no_grad():
        assert torch.equal(model(**batch, context=context, context_mask=torch.ones(2, 10, dtype=torch.bool)),
                           base(**batch))
    assert muon_parts("speaker_context.input.weight", model.speaker_context.input.weight) == 0  # raw latents in
    assert muon_parts("speaker_context.output.weight", model.speaker_context.output.weight) == 1


def test_context_conditions_rows_and_is_a_set_of_clips():
    torch.manual_seed(2)
    model = trained(FlowTTS(ModelConfig(**SMALL, **CONTEXT)))
    batch = inputs()
    a, b = torch.randn(2, 6, 4), torch.randn(2, 4, 4)
    full = torch.ones(2, 10, dtype=torch.bool)
    with torch.no_grad():
        ab = model(**batch, context=torch.cat([a, b], 1), context_mask=full)
        ba = model(**batch, context=torch.cat([b, a], 1), context_mask=full)
        assert torch.allclose(ab, ba, atol=1e-5)  # clip order does not matter (no positions; even clip lengths)
        assert not torch.allclose(ab, model(**batch))  # the context conditions the velocity
        garbage = torch.cat([a, b, 50 * torch.randn(2, 4, 4)], 1)  # masked padding does not leak
        mask = torch.arange(14)[None].expand(2, -1) < 10
        assert torch.allclose(model(**batch, context=garbage, context_mask=mask), ab, atol=1e-5)
        empty = torch.zeros(2, 10, dtype=torch.bool)  # no context rows add exactly nothing
        assert torch.equal(model(**batch, context=torch.randn(2, 10, 4), context_mask=empty), model(**batch))
        drop = torch.ones(2, dtype=torch.bool)  # condition dropout / CFG null drops it with the voice
        assert torch.equal(model(**batch, context=a, context_mask=full[:, :6], drop=drop), model(**batch, drop=drop))
        common = dict(prompt=batch["prompt"], prompt_mask=batch["prompt_mask"], valid=batch["valid"],
                      tokens=batch["tokens"], segments=batch["segments"], steps=3, guidance=2.0, seed=0)
        assert not torch.allclose(sample(model, **common, context=a, context_mask=full[:, :6]), sample(model, **common))
    with pytest.raises(ValueError, match="no speaker context"):
        FlowTTS(ModelConfig(**SMALL))(**batch, context=a, context_mask=full[:, :6])


def labels(cache):
    with sqlite3.connect(cache / "index.sqlite") as db:
        return dict(db.execute("SELECT uid, speaker FROM samples WHERE split='train'"))


def test_contexts_are_other_utterances_of_the_label_within_the_limits(cache):
    # The conftest cache: 24000 Hz / hop 512 = 46.875 fps, rows of 7/9/11 frames, 3 per speaker label.
    options = dict(speaker_context_prob=1.0, speaker_context_min_seconds=0.15, speaker_context_max_seconds=0.45)
    data = LatentDataset(cache, "train", 42, "within", "joined", **options)
    plain = LatentDataset(cache, "train", 42, "within", "joined")
    speaker, low, high = labels(cache), *data.context_frames
    lengths, drawn = {"-0": 7, "-1": 9, "-2": 11}, 0
    for epoch in range(3):
        for index in range(len(data)):
            item, base = data[(epoch, index)], plain[(epoch, index)]
            assert all(torch.equal(item[k], base[k]) for k in ("reference", "target"))  # nothing else moves
            assert data.context_plan(epoch, index) == data.context_plan(epoch, index)  # pure
            uids = [u for u in item["context_uid"].split("|") if u]
            assert item["uid"] not in uids and all(speaker[u] == speaker[item["uid"]] for u in uids)
            total = sum(lengths[u[-2:]] for u in uids)
            assert len(item["context"]) == total and (total == 0 or low <= total <= high)
            drawn += bool(uids)
    assert drawn > 0
    batch = collate([data[(0, i)] for i in range(4)])
    counts = [len(data[(0, i)]["context"]) for i in range(4)]
    assert batch["context"].shape[:2] == (4, max(1, max(counts))) and batch["context_mask"].sum(1).tolist() == counts
    assert "context" not in plain[(0, 0)] and "context" not in collate([plain[(0, 0)]])
    with pytest.raises(ValueError, match="speaker context"):
        collate([data[(0, 0)], plain[(0, 0)]])


def test_context_options_are_validated():
    model = ModelConfig(**SMALL, **CONTEXT)
    assert Config(model, TrainConfig(pairing="within", speaker_context_prob=0.5)).model.speaker_context == "vector"
    for bad in ((model, TrainConfig(pairing="within")), (ModelConfig(**SMALL), TrainConfig(speaker_context_prob=0.5,
                                                                                            pairing="within"))):
        with pytest.raises(ValueError, match="go together"):
            Config(*bad)
    for bad in (dict(speaker_context_prob=0.5), dict(speaker_context_prob=1.5, pairing="within"),
                dict(speaker_context_min_seconds=10, speaker_context_max_seconds=5)):
        with pytest.raises(ValueError):
            TrainConfig(**bad)
    with pytest.raises(ValueError):
        ModelConfig(**SMALL, speaker_context="tokens")
    with pytest.raises(ValueError):
        ModelConfig(**SMALL, **{**CONTEXT, "speaker_context_width": 9})


def test_a_context_branch_warm_starts_from_a_checkpoint_without_one():
    base_cfg, context_cfg = ModelConfig(**SMALL), ModelConfig(**SMALL, **CONTEXT)
    assert warm_start_config(base_cfg, context_cfg) == context_cfg  # only the context fields (and dropout) may differ
    assert warm_start_config(ModelConfig(**{**SMALL, "depth": 3}), context_cfg) != context_cfg
    base, model = FlowTTS(base_cfg), FlowTTS(context_cfg)
    warm_start(model, base.state_dict())  # the new branch stays zero-init: the model is the checkpoint's
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())
    with pytest.raises(ValueError, match="do not match"):
        warm_start(model, {k: v for k, v in base.state_dict().items() if not k.startswith("output")})
    with pytest.raises(ValueError, match="do not match"):
        warm_start(base, model.state_dict())  # dropping a trained branch is not a warm start


def test_inference_context_is_the_prompt_plus_whole_clips_up_to_the_trained_length():
    from dacvae_tts.inference import Synthesizer, VoiceReference

    tts = Synthesizer.__new__(Synthesizer)
    tts.model = FlowTTS(ModelConfig(**SMALL, **CONTEXT))
    tts.device, tts.mean, tts.std = torch.device("cpu"), torch.zeros(4), torch.ones(4)
    tts.codec = SimpleNamespace(sample_rate=2500, hop_length=100, latent_dim=4)
    tts.text_version, tts.duration_model = "turkish-v1", None
    tts.checkpoint = {"config": {"train": {"speaker_context_max_seconds": 1.0}}}  # 25 frames
    voice = VoiceReference(torch.randn(10, 4), "Önceki cümle.", "provided", {})
    clips = [torch.randn(8, 4), torch.randn(20, 4), torch.randn(5, 4)]
    context = tts.context_latents(voice, clips)  # 10 + 8 fit, 20 would pass 25 and is skipped, 5 more fit
    assert torch.equal(context, torch.cat([voice.latents, clips[0], clips[2]]))
    assert tts.context_latents(voice) is context  # kept on the reference
    batch = tts.make_batch(voice.latents, voice.transcript, "Yeni bir cümle.", context=context)
    assert batch["context"].shape == (1, 23, 4) and batch["context_mask"].all()
    alone = VoiceReference(torch.randn(10, 4), "Önceki cümle.", "provided", {})
    assert torch.equal(tts.context_latents(alone), alone.latents)  # no extra clips: the prompt itself
    plain = Synthesizer.__new__(Synthesizer)
    plain.model = FlowTTS(ModelConfig(**SMALL))
    assert plain.context_latents(voice) is None
