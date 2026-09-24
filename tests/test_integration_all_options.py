"""Integration of the feature branches (#2-#16, tracking issue #17): the options work together.

Every feature is config-gated and off by default; each branch pins its own "off == main" regression
(test_speed, test_latent_negatives, test_block_options, test_pairs, test_teacher,
test_schedule_regularization). This file covers what no single branch can: options of different
branches combined in one model, objective and training run, plus the combinations that config
validation rejects on purpose.
"""

import torch

from dacvae_tts.config import ModelConfig, TrainConfig
from dacvae_tts.data import collate
from dacvae_tts.model import FlowTTS
from dacvae_tts.speed import training_loader
from dacvae_tts.training import Objective


def tiny_batch(frames=(40, 52, 33), prompts=(6, 9, 5)):
    torch.manual_seed(0)
    items = []
    for index, (length, prompt) in enumerate(zip(frames, prompts)):
        latents = torch.randn(length, 4)
        items.append(
            dict(
                reference=latents[:prompt],
                target=latents[prompt:],
                reference_text="Önceki cümle burada.",
                text=f"Söylenen sözler numara {index} burada.",
            )
        )
    return collate(items)


def test_loader_negatives_are_text_hinge_only():
    """#7 x #8: loader-side text negatives are drawn for the transcript hinge only; latent_delta
    corrupts target latents inside the step, so the loader stays the plain collate."""

    class Rows:
        costs = [3, 4, 5]
        epoch = 0

        def __len__(self):
            return 3

    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2)
    assert training_loader(Rows(), train)[1].negatives
    train = TrainConfig(loader_negatives=True, contrastive_weight=0.2, contrastive_mode="latent_delta")
    items, collate_fn, _ = training_loader(Rows(), train)
    assert collate_fn is collate and isinstance(items, Rows)


def test_latent_delta_without_strict_checks_is_identical():
    """#7 x #8: strict_checks false only drops host-syncing value checks, latent negatives included."""
    cfg = ModelConfig(latent_dim=4, width=32, heads=2, depth=2, text_depth=1)
    results = []
    for strict in (True, False):
        torch.manual_seed(1)
        model = FlowTTS(cfg)
        torch.nn.init.normal_(model.output[-1].weight, std=0.05)
        model.strict_checks = strict
        objective = Objective(model, contrastive_mode="latent_delta").train()
        torch.manual_seed(2)
        results.append(objective(tiny_batch()))
    for key in ("flow", "latent_delta", "negative_random", "negative_aug"):
        assert torch.equal(results[0][key], results[1][key]), key


def test_utmos_flag_serves_protocol_and_eval_sentences():
    """#3 x #13: one `--utmos` switch. Bare it is the protocol's UTMOS22-strong (`utmos` field); with a
    model name (eval_sentences.py's former own flag) it picks that model, UTMOSv2 going to `utmosv2`."""
    import argparse

    from dacvae_tts.eval_protocol import add_protocol_args, protocol_from_args, utmos_models

    parser = argparse.ArgumentParser()
    add_protocol_args(parser)
    assert protocol_from_args(parser.parse_args([])) is None
    options = protocol_from_args(parser.parse_args(["--utmos"]))
    assert options.utmos and not options.utmosv2 and utmos_models(options) == ["utmos22"]
    options = protocol_from_args(parser.parse_args(["--utmos", "utmosv2"]))
    assert options.utmosv2 and not options.utmos and utmos_models(options) == ["utmosv2"]
    options = protocol_from_args(parser.parse_args(["--protocol-v2", "--utmos", "utmosv2"]))
    assert utmos_models(options) == ["utmos22", "utmosv2"]


class _Codec:
    """Codec stand-in: 4 latent channels, 512 samples per frame; accepts #13's decode keywords."""

    def __init__(self, checkpoint, device):
        self.latent_dim, self.sample_rate, self.hop_length = 4, 24000, 512
        self.metadata = dict(checkpoint="test-codec", sample_rate=24000, hop_length=512, latent_dim=4,
                             posterior="mean", weights_sha256="fixture", preprocessing="fixture")

    def decode(self, z, pre_tanh_gain=None, stats=None):
        if stats is not None:
            stats.update(pre_tanh_gain=1.0, pre_tanh_mode=str(pre_tanh_gain), pre_tanh_level=0.5)
        return torch.sin(z.sum(-1).cumsum(0)).repeat_interleave(512) * 0.3


def test_inference_options_combine(monkeypatch, cache, tmp_path):
    """#12 x #13: duration-diverse candidates, the articulation rule, a composite selector, two guidance
    windows and output shaping in one synthesize_many call."""
    import dacvae_tts.inference as module
    from dacvae_tts.config import Config
    from dacvae_tts.data import LatentDataset
    from dacvae_tts.inference import Synthesizer, VoiceReference

    data = LatentDataset(cache)
    config = Config(ModelConfig(latent_dim=4, width=16, depth=1, heads=2, text_depth=1, text_layout="joined",
                                duration="rule", positions="rope", prediction="edm"))
    path = tmp_path / "model.pt"
    model = FlowTTS(config.model)
    torch.save({"model": model.state_dict(), "ema": model.state_dict(), "config": config.to_dict(),
                "codec": {**data.meta, "text_normalization": "turkish-v1"}, "mean": data.mean, "std": data.std}, path)
    monkeypatch.setattr(module, "Codec", _Codec)
    tts = Synthesizer(path, device="cpu", precision="fp32")
    voice = VoiceReference(torch.randn(60, 4), "Referans cümlesi burada, biraz uzun.", "test", {})
    texts = ["Merhaba dünya.", "İkinci cümle biraz daha uzun."]

    def selector(text, audios, sample_rate):
        scores = [dict(dnsmos=float(len(audio))) for audio in audios]  # prefer the longest candidate
        return max(range(len(audios)), key=lambda k: scores[k]["dnsmos"]), scores

    results, metadata = tts.synthesize_many(
        texts, voice, candidates=3, steps=4, guidance=2.0, duration_mode="articulation",
        duration_factors=[1.0, 0.8, 1.25], selector=selector, guidance_split=0.5, apg_eta_late=0.5,
        moment_match="std", pre_tanh_gain="auto",
    )
    assert metadata["candidate_factors"] == [1.0, 0.8, 1.25] and metadata["moment_match"] == "std"
    assert metadata["late_window"]["eta"] == 0.5
    for row, best in zip(results, metadata["selected"]):
        assert [c["duration_factor"] for c in row] == [1.0, 0.8, 1.25]
        assert row[best]["duration_factor"] == 1.25 and "selection" in row[best] and "latent_moments" in row[best]
        assert all(torch.isfinite(torch.as_tensor(c["audio"])).all() for c in row)
