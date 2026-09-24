"""SIM-o (issue #3): checkpoint verification, key layout checks, input protocol and the transformers name mapping.

No WavLM weights are needed: a tiny s3prl-like stub stands in for the backbone. The transformers backbone itself was
checked bit-exact against the original WavLM code outside the test suite (transformers is not a test dependency).
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from dacvae_tts import sim_o
from dacvae_tts.codec import file_digest
from dacvae_tts.ecapa_tdnn import ECAPA_TDNN_SMALL


def speech_like(rate=16000, seconds=1.0, silence=1.0, seed=0):
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(rate * seconds)) / rate)
    tone += 0.05 * np.random.default_rng(seed).standard_normal(len(tone))
    return np.concatenate([tone, np.zeros(int(rate * silence))]).astype(np.float32)


class StubUpstream(torch.nn.Module):
    """s3prl-like extractor: list of 16 kHz waves -> 3 hidden states of width 1024 (320-sample frames)."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.proj = torch.nn.Linear(320, 1024)

    def forward(self, wavs):
        x = torch.stack(wavs)
        frames = x[:, : x.shape[1] // 320 * 320].reshape(x.shape[0], -1, 320)
        h0 = self.model.proj(frames)
        return {"hidden_states": [h0, torch.tanh(h0), 2 * h0]}


def fake_checkpoint(path, extra=None, drop=()):
    torch.manual_seed(3)
    model = ECAPA_TDNN_SMALL(StubUpstream(), feat_dim=1024, emb_dim=256)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    state = {k: v for k, v in model.state_dict().items() if k not in drop}
    state["loss_calculator.projection.weight"] = torch.randn(4, 256)
    state.update(extra or {})
    torch.save({"model": state, "best_valid_eer": 100.0}, path)
    return state


def test_sim_o_loader_verifies_sha256_and_key_layout(tmp_path):
    path = tmp_path / "wavlm_large_finetune.pth"
    state = fake_checkpoint(path)
    digest = file_digest(path)
    expected = len(state) - 1  # everything but the training classifier
    scorer = sim_o.SimO(path, backbone=StubUpstream(), expected_sha256=digest, min_matched=expected)
    assert scorer.matched == expected and scorer.identity["backend"] == "custom"
    assert scorer.identity["sha256"] == digest and scorer.identity["repo"] is None
    loaded = scorer.model.state_dict()
    assert all(torch.equal(loaded[k].cpu(), v) for k, v in state.items() if not k.startswith("loss_calculator"))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        sim_o.SimO(path, backbone=StubUpstream(), min_matched=expected)  # the real hash, a different file
    with pytest.raises(ValueError, match=f"need >= {expected + 1}"):
        sim_o.SimO(path, backbone=StubUpstream(), expected_sha256=digest, min_matched=expected + 1)
    assert sim_o.MIN_MATCHED_KEYS == 700 and sim_o.SIZE == 1_301_926_579
    with pytest.raises(FileNotFoundError):
        sim_o.resolve_checkpoint(tmp_path / "missing.pth")


def test_sim_o_loader_rejects_stray_and_missing_tensors(tmp_path):
    stray = tmp_path / "stray.pth"
    fake_checkpoint(stray, extra={"feature_extract.model.renamed.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="Unexpected"):
        sim_o.SimO(stray, backbone=StubUpstream(), verify_sha256=False, min_matched=10)
    missing = tmp_path / "missing.pth"
    fake_checkpoint(missing, drop=("linear.weight",))
    with pytest.raises(ValueError, match="missing"):
        sim_o.SimO(missing, backbone=StubUpstream(), verify_sha256=False, min_matched=10)
    torch.save({"state_dict": {}}, tmp_path / "bad.pth")
    with pytest.raises(ValueError, match="'model'"):
        sim_o.load_state(tmp_path / "bad.pth")
    with pytest.raises(ValueError, match="backend"):
        sim_o.SimO(missing, backend="fairseq", verify_sha256=False)


def test_sim_o_similarity_uses_first_channel_at_16k(tmp_path):
    path = tmp_path / "ckpt.pth"
    fake_checkpoint(path)
    scorer = sim_o.SimO(path, backbone=StubUpstream(), verify_sha256=False, min_matched=10)
    voice = speech_like(48000, silence=0.2)
    sf.write(tmp_path / "a.wav", np.stack([voice, np.zeros_like(voice)], 1), 48000)
    sf.write(tmp_path / "b.wav", voice, 48000)
    other = np.random.default_rng(5).standard_normal(len(voice)).astype(np.float32) * 0.3
    sf.write(tmp_path / "c.wav", other, 48000)
    wave = sim_o.load_audio_16k(tmp_path / "a.wav")
    assert wave.shape == (19200,)  # resampled, first channel only (the silent second channel is ignored)
    assert scorer.similarity(tmp_path / "a.wav", tmp_path / "b.wav") == pytest.approx(1.0, abs=1e-5)
    assert scorer.similarity(tmp_path / "a.wav", tmp_path / "c.wav") < 0.999
    assert len(scorer._cache) == 2  # prompt embeddings are cached


def test_wavlm_names_map_one_to_one_onto_transformers():
    names = ["mask_emb", "layer_norm.weight", "layer_norm.bias", "post_extract_proj.weight", "post_extract_proj.bias",
             "encoder.pos_conv.0.bias", "encoder.pos_conv.0.weight_g", "encoder.pos_conv.0.weight_v",
             "encoder.layer_norm.weight", "encoder.layer_norm.bias",
             "encoder.layers.0.self_attn.relative_attention_bias.weight"]
    for i in range(7):
        names += [f"feature_extractor.conv_layers.{i}.0.weight", f"feature_extractor.conv_layers.{i}.2.1.weight",
                  f"feature_extractor.conv_layers.{i}.2.1.bias"]
    for i in range(24):
        for part in ("k_proj", "v_proj", "q_proj", "out_proj", "grep_linear"):
            names += [f"encoder.layers.{i}.self_attn.{part}.weight", f"encoder.layers.{i}.self_attn.{part}.bias"]
        names.append(f"encoder.layers.{i}.self_attn.grep_a")
        for part in ("self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"):
            names += [f"encoder.layers.{i}.{part}.weight", f"encoder.layers.{i}.{part}.bias"]
    assert len(names) == 488  # the WavLM-Large tensors of s3prl's wavlm_large.pt and of the SIM-o checkpoint
    mapped = [sim_o.wavlm_to_transformers(name) for name in names]
    assert len(set(mapped)) == len(names)
    expected = {
        "mask_emb": "masked_spec_embed",
        "feature_extractor.conv_layers.3.0.weight": "feature_extractor.conv_layers.3.conv.weight",
        "feature_extractor.conv_layers.3.2.1.bias": "feature_extractor.conv_layers.3.layer_norm.bias",
        "layer_norm.weight": "feature_projection.layer_norm.weight",
        "post_extract_proj.bias": "feature_projection.projection.bias",
        "encoder.pos_conv.0.weight_g": "encoder.pos_conv_embed.conv.weight_g",
        "encoder.layers.0.self_attn.relative_attention_bias.weight": "encoder.layers.0.attention.rel_attn_embed.weight",
        "encoder.layers.5.self_attn.grep_a": "encoder.layers.5.attention.gru_rel_pos_const",
        "encoder.layers.5.self_attn.grep_linear.weight": "encoder.layers.5.attention.gru_rel_pos_linear.weight",
        "encoder.layers.5.self_attn.q_proj.bias": "encoder.layers.5.attention.q_proj.bias",
        "encoder.layers.5.self_attn_layer_norm.weight": "encoder.layers.5.layer_norm.weight",
        "encoder.layers.5.fc1.weight": "encoder.layers.5.feed_forward.intermediate_dense.weight",
        "encoder.layers.5.fc2.bias": "encoder.layers.5.feed_forward.output_dense.bias",
        "encoder.layers.5.final_layer_norm.bias": "encoder.layers.5.final_layer_norm.bias",
        "encoder.layer_norm.bias": "encoder.layer_norm.bias",
    }
    for name, target in expected.items():
        assert sim_o.wavlm_to_transformers(name) == target
    parametrized = ["encoder.pos_conv_embed.conv.parametrizations.weight.original0",
                    "encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
    assert sim_o.wavlm_to_transformers("encoder.pos_conv.0.weight_g", parametrized) == parametrized[0]
    assert sim_o.wavlm_to_transformers("encoder.pos_conv.0.weight_v", parametrized) == parametrized[1]
    with pytest.raises(KeyError):
        sim_o.wavlm_to_transformers("encoder.layers.0.adapter.weight")


def test_remap_state_renames_only_the_backbone():
    prefix = sim_o.BACKBONE_PREFIX
    targets = {prefix + "masked_spec_embed": 0, prefix + "encoder.pos_conv_embed.conv.parametrizations.weight.original0": 0,
               "linear.weight": 0}
    model = type("FakeModel", (), {"state_dict": lambda self: targets})()
    state = {prefix + "mask_emb": 1, prefix + "encoder.pos_conv.0.weight_g": 2, "linear.weight": 3,
             "loss_calculator.projection.weight": 4}
    assert sim_o.remap_state(state, model) == {
        prefix + "masked_spec_embed": 1, prefix + "encoder.pos_conv_embed.conv.parametrizations.weight.original0": 2,
        "linear.weight": 3, "loss_calculator.projection.weight": 4,
    }
    with pytest.raises(KeyError, match="Unmapped"):  # a tensor the mapping does not know is an error, not dropped
        sim_o.remap_state({prefix + "quantizer.vars": 1}, model)
