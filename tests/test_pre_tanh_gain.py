"""Pre-tanh gain of `Codec.decode` (issue 13) on a DACVAE-like fake decoder head."""

import math

import pytest
import torch
from torch import nn

from dacvae_tts.codec import (
    Codec,
    output_tanh,
    parse_pre_tanh_gain,
    pre_tanh_gain_hook,
    pre_tanh_mode,
    robust_level,
)


def snake(x, alpha):
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).square()


class WatermarkedDecoder(nn.Module):
    """DACVAE-like head: output = tanh(conv(snake(x))) + alpha * watermark(tanh(...)), the tanh called twice."""

    def __init__(self, channels=4, scale=8.0, tanh=True, extra_tanh=False, use_tanh=True):
        super().__init__()
        torch.manual_seed(11)
        self.snake_alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.conv = nn.Conv1d(channels, 1, 7, padding=3)
        self.pre = nn.Sequential(nn.Tanh() if tanh else nn.Identity(), nn.Conv1d(1, 2, 7, padding=3))
        self.post = nn.Sequential(nn.ELU(), nn.Conv1d(2, 1, 1), nn.Tanh() if extra_tanh else nn.Identity())
        self.use_tanh = use_tanh
        with torch.no_grad():
            self.conv.weight.mul_(scale)
            self.post[1].weight.mul_(0.05)  # an inaudible watermark residual, as in DACVAE
            self.post[1].bias.zero_()

    def forward(self, z):
        head = self.conv(snake(z, self.snake_alpha))
        if not self.use_tanh:
            return head
        watermark = self.post(self.pre(head))  # the watermark branch sees the tanh first
        return self.pre[0](head) + 0.25 * watermark  # then the output path (forward_no_conv)


class FakeDACVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder = WatermarkedDecoder(**kwargs)

    def decode(self, z):
        return self.decoder(z)


def fake_codec(fast=False, **kwargs):
    codec = Codec.__new__(Codec)
    codec.model = FakeDACVAE(**kwargs).eval()
    codec.latent_dim, codec.device, codec.encoder_only = 4, torch.device("cpu"), False

    class Fast:  # FastCodec.decode runs the original decoder head (watermark network) of `model`
        def decode(self, latent, model):
            return model.decoder(latent)

    codec._fast = Fast() if fast else None
    return codec


def latents(frames=4000, seed=0):
    return torch.randn(frames, 4, generator=torch.Generator().manual_seed(seed))


def test_decode_without_gain_is_unchanged():
    codec, z = fake_codec(), latents()
    direct = codec.model.decode(z.T[None].contiguous().float())[0, 0]
    assert torch.equal(codec.decode(z), direct)
    stats = {}
    assert torch.equal(codec.decode(z, pre_tanh_gain=None, stats=stats), direct) and stats == {}
    assert not output_tanh(codec.model.decoder)._forward_pre_hooks


@pytest.mark.parametrize("fast", [False, True])
def test_auto_gain_removes_saturation_consistently_in_one_pass(fast):
    codec, z = fake_codec(fast=fast), latents()
    tanh = output_tanh(codec.model.decoder)
    plain = codec.decode(z)
    assert float((plain.abs() >= 0.999).float().mean()) > 0.05  # the fake head saturates like high-CFG outputs
    seen = []
    handle = tanh.register_forward_hook(lambda module, args, output: seen.append(args[0].clone()))
    stats = {}
    fixed = codec.decode(z, pre_tanh_gain="auto", stats=stats)
    handle.remove()
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])  # watermark branch and output: same scaled input
    assert stats["pre_tanh_mode"] == "auto" and 0 < stats["pre_tanh_gain"] < 1
    assert stats["pre_tanh_level"] > math.atanh(0.95)
    assert robust_level(seen[0]) == pytest.approx(math.atanh(0.95), rel=1e-5)  # 99.9th percentile -> atanh(0.95)
    assert float((fixed.abs() >= 0.999).float().mean()) < 0.002
    assert not tanh._forward_pre_hooks
    stricter = {}
    codec.decode(z, pre_tanh_gain="auto:0.8", stats=stricter)
    assert stricter["pre_tanh_gain"] < stats["pre_tanh_gain"]


def test_fixed_gain_equals_scaling_the_last_convolution():
    codec, z = fake_codec(), latents()
    stats = {}
    halved = codec.decode(z, pre_tanh_gain=0.5, stats=stats)
    assert stats["pre_tanh_gain"] == 0.5 and stats["pre_tanh_level"] is None
    with torch.no_grad():
        codec.model.decoder.conv.weight.mul_(0.5)
        codec.model.decoder.conv.bias.mul_(0.5)
    torch.testing.assert_close(halved, codec.decode(z), atol=1e-5, rtol=1e-5)


def test_auto_gain_is_the_identity_on_unsaturated_audio():
    codec, z = fake_codec(scale=0.05), latents()
    stats = {}
    assert torch.equal(codec.decode(z, pre_tanh_gain="auto", stats=stats), codec.decode(z))
    assert stats["pre_tanh_gain"] == 1.0


def test_gain_errors_and_parsing():
    z = latents(50)
    with pytest.raises(ValueError, match="no nn.Tanh"):
        fake_codec(tanh=False).decode(z, pre_tanh_gain=0.5)
    with pytest.raises(ValueError, match="ambiguous"):
        fake_codec(extra_tanh=True).decode(z, pre_tanh_gain="auto")
    with pytest.raises(RuntimeError, match="never reached"):
        fake_codec(use_tanh=False).decode(z, pre_tanh_gain=0.5)
    codec = fake_codec()
    for bad in (0, -1.0, float("inf"), "auto:1.5", "auto:x", "loud", True):
        with pytest.raises(ValueError):
            codec.decode(z, pre_tanh_gain=bad)
    assert not output_tanh(codec.model.decoder)._forward_pre_hooks  # removed after failures too
    assert parse_pre_tanh_gain("AUTO") == "auto" and parse_pre_tanh_gain("auto:0.9") == "auto:0.9"
    assert parse_pre_tanh_gain("0.7") == 0.7 and parse_pre_tanh_gain(None) is None
    assert pre_tanh_mode("auto") == ("auto", 0.95) and pre_tanh_mode(2) == ("fixed", 2.0)
    with pytest.raises(ValueError):
        parse_pre_tanh_gain("-2")
    decoder = WatermarkedDecoder()
    with pytest.raises(RuntimeError):
        with pre_tanh_gain_hook(decoder, 0.5):
            raise RuntimeError("boom")
    assert not output_tanh(decoder)._forward_pre_hooks
