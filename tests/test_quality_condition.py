"""Quality condition (model.quality_condition): model, data, config, optimizer, inference defaults."""

import json

import pytest
import torch

from dacvae_tts.config import Config, ModelConfig, TrainConfig
from dacvae_tts.data import LatentDataset, collate
from dacvae_tts.model import FlowTTS, quality_features, sample
from dacvae_tts.optim import muon_parts
from dacvae_tts.text import tokenize

SMALL = dict(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined", duration="rule",
             positions="rope", prediction="edm", adaln_rank=4)


def inputs(batch=2, frames=9, prompt=4):
    tokens, segments = tokenize("Önceki cümle.", "Yeni bir cümle.", "turkish-v1", "joined")
    x = torch.randn(batch, frames, 4)
    mask = torch.arange(frames)[None].expand(batch, -1) < prompt
    return dict(x=x, time=torch.full((batch,), 0.4), prompt=x * mask[..., None], prompt_mask=mask,
                valid=torch.ones(batch, frames, dtype=torch.bool), tokens=tokens[None].repeat(batch, 1),
                segments=segments[None].repeat(batch, 1))


def test_off_by_default_and_zero_init_equals_the_baseline():
    assert ModelConfig().quality_condition is False
    torch.manual_seed(0)
    base = FlowTTS(ModelConfig(**SMALL))
    torch.manual_seed(0)
    model = FlowTTS(ModelConfig(**SMALL, quality_condition=True))
    assert set(model.state_dict()) - set(base.state_dict()) == {"quality_condition.weight"}
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())  # no extra draws
    batch = inputs()
    with torch.no_grad():
        expected = base(**batch)
        for quality in (None, torch.zeros(2, 3), torch.randn(2, 3)):
            assert torch.equal(model(**batch, quality=quality), expected)
    assert muon_parts("quality_condition.weight", model.quality_condition.weight) == 0  # AdamW boundary


def test_quality_moves_the_prediction_and_the_null_branch_drops_it():
    torch.manual_seed(1)
    model = FlowTTS(ModelConfig(**SMALL, quality_condition=True)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    batch = inputs()
    high, low = quality_features([[3.8, 4.2, 3.5]] * 2), quality_features([[2.0, 2.2, 1.6]] * 2)
    with torch.no_grad():
        assert not torch.allclose(model(**batch, quality=high), model(**batch, quality=low))
        drop = torch.ones(2, dtype=torch.bool)
        assert torch.equal(model(**batch, quality=high, drop=drop), model(**batch, quality=low, drop=drop))
        common = dict(prompt=batch["prompt"], prompt_mask=batch["prompt_mask"], valid=batch["valid"],
                      tokens=batch["tokens"], segments=batch["segments"], steps=3, guidance=2.0, seed=0)
        assert not torch.allclose(sample(model, **common, quality=high), sample(model, **common, quality=low))
    with pytest.raises(ValueError, match="Quality must be"):
        model(**batch, quality=torch.zeros(2, 2))


def test_features_center_scale_and_unknown():
    assert torch.equal(quality_features([3.0, 3.5, float("nan")]), torch.tensor([0.0, 1.0, 0.0]))


def test_dataset_carries_the_target_quality_with_dropout(cache):
    scores = {f"train-{s}-{u}": [3.0 + 0.1 * s, 3.5, 2.5 + 0.1 * u] for s in range(4) for u in range(3)}
    del scores["train-0-0"]  # unscored row -> unknown (zeros)
    (cache / "quality.json").write_text(json.dumps(scores))
    plain = LatentDataset(cache, "train", pairing="within", layout="joined")
    data = LatentDataset(cache, "train", pairing="within", layout="joined", quality_scores=cache / "quality.json",
                         quality_dropout=0.0)
    for index in range(len(data)):
        item, reference = data[index], plain[index]
        assert torch.equal(item["target"], reference["target"])  # the condition moves no other draw
        expected = scores.get(item["uid"])
        want = torch.zeros(3) if expected is None else quality_features(expected)
        assert torch.allclose(item["quality"], want)
    batch = collate([data[i] for i in range(4)])
    assert batch["quality"].shape == (4, 3)
    dropped = LatentDataset(cache, "train", pairing="within", layout="joined", quality_scores=cache / "quality.json",
                            quality_dropout=1.0)
    assert all(torch.equal(dropped[i]["quality"], torch.zeros(3)) for i in range(len(dropped)))
    assert "quality" not in collate([plain[0], plain[1]])


def test_config_pairs_the_model_flag_with_a_store():
    with pytest.raises(ValueError, match="go together"):
        Config(ModelConfig(**SMALL, quality_condition=True), TrainConfig(pairing="within"))
    with pytest.raises(ValueError, match="quality_target"):
        ModelConfig(**SMALL, quality_target=(3.0, 3.0))
    config = Config(ModelConfig(**SMALL, quality_condition=True, quality_target=[3.5, 4.0, 3.2]),
                    TrainConfig(pairing="within", quality_scores="quality/dnsmos.json"))
    assert Config.from_dict(config.to_dict()).to_dict() == config.to_dict()  # resume compares these


def test_synthesizer_generate_asks_for_the_quality_target():
    """Synthesizer.generate builds its own condition cache: the requested quality must be in it (review finding)."""
    from types import SimpleNamespace

    from dacvae_tts.inference import Synthesizer

    torch.manual_seed(2)
    model = FlowTTS(ModelConfig(**SMALL, quality_condition=True, quality_target=(3.9, 4.2, 3.6))).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    tts = Synthesizer.__new__(Synthesizer)
    tts.model, tts.device, tts.precision, tts.profile = model, torch.device("cpu"), "fp32", False
    tts.quality_target = None
    captured = {}

    def finish(target, prompt, **options):
        captured["target"] = target.clone()
        return torch.zeros(10), {}

    tts._finish = finish
    tts.codec = SimpleNamespace(sample_rate=2500, hop_length=100, latent_dim=4)
    batch = inputs(batch=1)
    batch = {k: v for k, v in batch.items() if k not in ("x", "time")}
    try:
        tts.generate(dict(batch), steps=3, guidance=2.0, seed=0)
    except Exception:  # metadata after decoding needs a real codec; the latents are captured before that
        pass
    core = {k: batch[k] for k in ("prompt", "prompt_mask", "valid", "tokens", "segments")}
    mask = batch["valid"][0] & ~batch["prompt_mask"][0]
    wanted = sample(model, **core, steps=3, guidance=2.0, seed=0, quality=quality_features([[3.9, 4.2, 3.6]]))
    unknown = sample(model, **core, steps=3, guidance=2.0, seed=0, quality=torch.zeros(1, 3))
    assert torch.allclose(captured["target"], wanted[0, mask], atol=1e-5)
    assert not torch.allclose(captured["target"], unknown[0, mask], atol=1e-3)
