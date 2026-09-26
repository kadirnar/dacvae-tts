"""Fixes from the 26 September architecture audit: weight-decay scope, TLA-SA on prompt-free rows, the speaker
condition's inference audio and exponent overrides in the A/B runner."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dacvae_tts.config import TrainConfig
from dacvae_tts.optim import build_optimizer, decay_exempt


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 4)
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.gate = nn.Parameter(torch.ones(4))


@pytest.mark.parametrize("name", ["muon", "adamw"])
def test_matrices_scope_leaves_vectors_and_embeddings_undecayed(name):
    model = Toy()
    exempt = decay_exempt(model)
    assert id(model.proj.weight) not in exempt
    assert {id(model.embed.weight), id(model.proj.bias), id(model.norm.weight), id(model.gate)} <= exempt
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = build_optimizer(model, name, lr=0.1, weight_decay=0.5, decay_scope="matrices")
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()  # zero gradients: only weight decay moves parameters
    after = dict(model.named_parameters())
    for n in ("embed.weight", "proj.bias", "norm.weight", "norm.bias", "gate"):
        assert torch.equal(after[n], before[n]), n
    assert torch.allclose(after["proj.weight"], before["proj.weight"] * (1 - 0.1 * 0.5))


def test_all_scope_is_the_previous_recipe():
    model = Toy()
    optimizer = build_optimizer(model, "muon", lr=0.1, weight_decay=0.5)
    assert len(optimizer.param_groups) == 2 and all(g["weight_decay"] == 0.5 for g in optimizer.param_groups)
    with pytest.raises(ValueError):
        TrainConfig(weight_decay_scope="vectors")


def test_tla_ignores_rows_without_prompt_frames():
    from dacvae_tts.alignment import teacher_terms
    from dacvae_tts.model import FlowTTS, ModelConfig

    cfg = ModelConfig(latent_dim=4, width=16, heads=2, depth=2, text_depth=1, text_layout="joined",
                      duration="rule", tla_layers="all", tla_dim=3, tla_hidden=8, patch_size=1)
    model = FlowTTS(cfg)
    b, length = 3, 10
    valid = torch.ones(b, length, dtype=torch.bool)
    prompt_mask = torch.zeros(b, length, dtype=torch.bool)
    prompt_mask[0, :4] = True  # row 0 has a prompt, rows 1 and 2 do not (prompt dropout)
    hidden = {layer: torch.randn(b, length, 16) for layer in range(1, 3)}
    batch = {"valid": valid, "prompt_mask": prompt_mask, "speaker_embedding": torch.randn(b, 3)}
    drop = torch.tensor([False, False, True])
    terms = teacher_terms(model, batch, hidden, torch.rand(b), drop, repa=False, tla=True)
    assert terms["tla"][0] != 0 and (terms["tla"][1:] == 0).all() and (terms["tla_entropy"][1:] == 0).all()


def test_speaker_condition_embeds_the_codec_input_when_the_store_used_original_audio():
    from dacvae_tts.inference import Synthesizer, VoiceReference

    seen = []

    def embedder(audio, rate):
        seen.append(np.asarray(audio).copy())
        return torch.ones(3)

    tts = Synthesizer.__new__(Synthesizer)
    tts.model = SimpleNamespace(speaker_condition=object())
    tts.device, tts.mean, tts.std, tts._speaker_embedder = torch.device("cpu"), torch.zeros(4), torch.ones(4), embedder
    tts.codec = SimpleNamespace(sample_rate=100, loudness=None, decode=lambda latents: torch.full((50,), 7.0))
    waveform = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    tts.speaker_record = {"audio_sources": {"original": 10, "decoded": 0}}
    tts.speaker_embedding(VoiceReference(torch.zeros(5, 4), "a", "provided", {}, audio=(waveform, 100)))
    assert np.allclose(seen[-1], waveform)  # the codec's input, not its reconstruction
    tts.speaker_record = {"audio_sources": {"original": 0, "decoded": 10}}
    tts.speaker_embedding(VoiceReference(torch.zeros(5, 4), "a", "provided", {}, audio=(waveform, 100)))
    assert np.allclose(seen[-1], 7.0)  # a store of decoded latents: the reconstruction, as before
    tts.speaker_record = {"audio_sources": {"original": 10, "decoded": 0}}
    tts.speaker_embedding(VoiceReference(torch.zeros(5, 4), "a", "cache", {}))
    assert np.allclose(seen[-1], 7.0)  # built from cached latents: no audio to embed


def test_runner_reads_exponent_overrides_as_numbers(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "trc"))
    import yaml
    from run_arm import derive

    out = derive("configs/nano_tr_w512_fast.yaml", tmp_path / "c.yaml", ["train.learning_rate=1e-4"])
    assert yaml.safe_load(out.read_text())["train"]["learning_rate"] == 1e-4
