"""Reference-conditioned, non-autoregressive continuous latent flow model."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .contracts import audio_shapes, mask_values, sanitize, text_shapes
from .reference import ReferencePool, TemporalReference
from .text import BYTE_OFFSET, VOCAB_SIZE


def sinusoidal(positions, width):
    freq = torch.exp(
        torch.arange(width // 2, device=positions.device).float()
        * (-math.log(10000) / max(width // 2 - 1, 1))
    )
    angles = positions.float().unsqueeze(-1) * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def masked_mean(x, mask):
    return sanitize(x, mask).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)

    def forward(self, x, context, valid):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.embedding = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.segment = nn.Embedding(2, d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 7, padding=3, groups=d) for _ in range(cfg.text_depth)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                for _ in range(cfg.text_depth)
            ]
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, segments):
        valid = tokens.ne(0)
        x = self.embedding(tokens) + self.segment(segments)
        x = x + sinusoidal(torch.arange(tokens.size(1), device=x.device), x.size(-1)).to(x.dtype)
        for conv, mlp in zip(self.convs, self.mlps):
            x = x * valid[..., None]
            x = x + mlp(conv(x.transpose(1, 2)).transpose(1, 2))
        return self.norm(x) * valid[..., None], valid


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = Attention(d, cfg.heads)
        self.cross_attn = Attention(d, cfg.heads)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg.ff_mult), nn.GELU(approximate="tanh"), nn.Linear(d * cfg.ff_mult, d)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 9))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, text, valid, text_valid, cond):
        params = self.ada(cond).unsqueeze(1).chunk(9, dim=-1)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = params
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.self_attn(h, h, valid)
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, text, text_valid)
        x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        return x * valid[..., None]


class FlowTTS(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, c, p = cfg.width, cfg.latent_dim, cfg.patch_size
        self.text = TextEncoder(cfg)
        self.ref = nn.Sequential(nn.Linear(c, d), nn.SiLU(), nn.Linear(d, d))
        if cfg.reference_encoder == "temporal":
            self.ref = TemporalReference(c, d)
        self.ref_pool = ReferencePool(d, cfg.reference_pooling) if cfg.reference_pooling != "mean" else None
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.input = nn.Linear((2 * c + 1) * p, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c * p))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        duration_extra = 3 if cfg.duration_features == "text_stats" else 0
        self.duration = nn.Sequential(nn.Linear(2 * d + 1 + duration_extra, d), nn.SiLU(), nn.Linear(d, 1))
        self.grad_checkpoint = False

    def reference_summary(self, prompt, prompt_mask):
        prompt = sanitize(prompt, prompt_mask)
        features = (
            self.ref(prompt, prompt_mask) if self.cfg.reference_encoder == "temporal" else self.ref(prompt)
        )
        voice = (
            self.ref_pool(features, prompt_mask)
            if self.ref_pool is not None
            else masked_mean(features, prompt_mask)
        )
        if self.cfg.reference_paths == "full":
            voice = voice * 0  # Keep a connected graph for DDP; this path is intentionally inactive.
        return voice

    def conditions(self, prompt, prompt_mask, tokens, segments, drop=None):
        text, text_valid = self.text(tokens, segments)
        voice = self.reference_summary(prompt, prompt_mask)
        if drop is not None:
            text = text.masked_fill(drop[:, None, None], 0)
            voice = voice.masked_fill(drop[:, None], 0)
        return text, text_valid, voice

    def predict_duration(self, prompt, prompt_mask, tokens, segments, cached=None):
        if (
            prompt.ndim != 3
            or prompt.shape[-1] != self.cfg.latent_dim
            or prompt_mask.shape != prompt.shape[:2]
        ):
            raise ValueError("Duration prompt must be [B,L,C] with matching [B,L] reference mask")
        text_shapes(tokens, segments, prompt.size(0), prompt.device)
        text, _, voice = self.conditions(prompt, prompt_mask, tokens, segments) if cached is None else cached
        target_bytes = (segments == 1) & (tokens >= BYTE_OFFSET)
        if (target_bytes.sum(1) == 0).any():
            raise ValueError("Duration prediction requires nonempty target text")
        ref_bytes = ((segments == 0) & (tokens >= BYTE_OFFSET)).sum(1).clamp_min(1)
        rate = (prompt_mask.sum(1).float().clamp_min(1) / ref_bytes).log().unsqueeze(1)
        features = [masked_mean(text, target_bytes), voice, rate]
        if self.cfg.duration_features == "text_stats":
            byte_values = tokens - BYTE_OFFSET
            characters = (target_bytes & ((byte_values & 0xC0) != 0x80)).sum(1).float().clamp_min(1)
            punctuation = torch.zeros_like(tokens, dtype=torch.bool)
            for value in b".,;:!?":
                punctuation |= byte_values == value
            punct_fraction = (punctuation & target_bytes).sum(1) / characters
            spaces = (byte_values == 32) & target_bytes
            words = (spaces.sum(1) + 1).float()
            features.append(torch.stack([characters.log(), punct_fraction, words.log()], -1))
        return self.duration(torch.cat(features, -1)).squeeze(-1)

    def forward(self, x, time, prompt, prompt_mask, valid, tokens, segments, drop=None, cached=None):
        audio_shapes(x, prompt, prompt_mask, valid, self.cfg.latent_dim)
        text_shapes(tokens, segments, x.size(0), x.device)
        if time.shape != (x.size(0),) or time.device != x.device or not time.is_floating_point():
            raise ValueError("Flow time must be floating [B] on the audio device")
        if drop is not None and (
            drop.shape != time.shape or drop.dtype != torch.bool or drop.device != x.device
        ):
            raise ValueError("Condition dropout must be boolean [B] on the audio device")
        prompt_mask = prompt_mask & valid
        x, prompt = sanitize(x, valid), sanitize(prompt, prompt_mask)
        b, length, channels = x.shape
        p = self.cfg.patch_size
        if cached is None:
            cached = self.conditions(prompt, prompt_mask, tokens, segments, drop)
        text, text_valid, voice = cached
        if (
            text.shape != (*tokens.shape, self.cfg.width)
            or text_valid.shape != tokens.shape
            or voice.shape != (b, self.cfg.width)
        ):
            raise ValueError("Cached text must be [B,S,D], text mask [B,S], reference summary [B,D]")
        if drop is not None:
            # Centralize payload removal so direct calls and training/sampling agree.
            x = x.masked_fill((drop[:, None] & prompt_mask)[..., None], 0)
            prompt = prompt.masked_fill(drop[:, None, None], 0)
            text = text.masked_fill(drop[:, None, None], 0)
            voice = voice.masked_fill(drop[:, None], 0)
            prompt_mask = prompt_mask & (~drop)[:, None]
        if self.cfg.reference_paths == "summary":
            x = x.masked_fill(prompt_mask[..., None], 0)
            prompt = torch.zeros_like(prompt)
        features = torch.cat([x, prompt, prompt_mask[..., None].to(x.dtype)], -1)
        pad = (-length) % p
        features = F.pad(features, (0, 0, 0, pad))
        packed_valid = F.pad(valid, (0, pad)).reshape(b, -1, p).any(-1)
        if self.input.in_features != p * (2 * channels + 1) or self.output[-1].out_features != p * channels:
            raise ValueError("Invalid packing projections: expected P(2C+1) input and PC output")
        h = self.input(features.reshape(b, -1, features.size(-1) * p))
        h = h + sinusoidal(torch.arange(h.size(1), device=x.device), h.size(-1)).to(h.dtype)
        time_embedding = self.time(sinusoidal(time * 1000, self.cfg.width).to(h.dtype))
        if time_embedding.shape != voice.shape:
            raise ValueError("Time embedding and reference summary must both be [B,D]")
        cond = time_embedding + voice
        for block in self.blocks:
            args = (h, text, packed_valid, text_valid, cond)
            h = (
                checkpoint(block, *args, use_reentrant=False)
                if self.grad_checkpoint and self.training
                else block(*args)
            )
        output = self.output(h)
        expected_packs = (length + pad) // p
        if output.shape != (b, expected_packs, channels * p):
            raise ValueError("Velocity projection returned an unexpected packed length or width")
        # Remove only the explicitly added packing padding, then mask at frame resolution.
        return sanitize(output.reshape(b, length + pad, channels)[:, :length], valid)


def per_example_mse(prediction, target, mask):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or mask.shape != prediction.shape[:2]
        or mask.dtype != torch.bool
    ):
        raise ValueError("MSE requires matching [B,L,C] predictions/targets and boolean [B,L] mask")
    if (mask.sum(-1) == 0).any():
        raise ValueError("Each example must contain at least one valid target frame")
    error = (sanitize(prediction.float(), mask) - sanitize(target.float(), mask)).square().mean(-1)
    return error.sum(-1) / mask.sum(-1)


def reduce_flow(losses, counts, reduction="utterance"):
    if reduction == "utterance":
        return losses.mean()
    if reduction == "frame":
        return (losses * counts).sum() / counts.sum()
    raise ValueError("reduction must be utterance or frame")


def flow_loss(model, batch, dropout=0.1, time=None, noise=None, return_details=False):
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must lie in [0,1]")
    mask = mask_values(batch["valid"], batch["prompt_mask"])
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    time = torch.rand(b, device=x1.device) if time is None else time
    noise = torch.randn_like(x1) if noise is None else noise
    if time.shape != (b,) or noise.shape != x1.shape or time.device != x1.device or noise.device != x1.device:
        raise ValueError("Noise must match [B,L,C]; time must match [B], on the audio device")
    if not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
        raise ValueError("Flow time must be finite and lie in [0,1]")
    noise = sanitize(noise, batch["valid"])
    xt = (1 - time[:, None, None]) * noise + time[:, None, None] * x1
    # The conditioning prefix follows exactly the same path at training and inference.
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    drop = torch.rand(b, device=x1.device) < dropout
    # Remove reference from the state too, otherwise CFG's null branch leaks the voice.
    xt = xt.masked_fill((drop[:, None] & batch["prompt_mask"])[..., None], 0)
    pred = model(
        xt,
        time,
        batch["prompt"],
        batch["prompt_mask"],
        batch["valid"],
        batch["tokens"],
        batch["segments"],
        drop=drop,
    )
    losses = per_example_mse(pred, x1 - noise, mask)
    if return_details:
        counts = mask.sum(1)
        rms = (sanitize(pred.float(), mask).square().sum((1, 2)) / (counts * pred.size(-1))).sqrt()
        return {"flow": losses, "times": time, "frames": counts, "prediction_rms": rms}
    return losses


def time_grid(steps, sway, device, times=None):
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 1
        or not math.isfinite(sway)
        or not -1 <= sway <= 0
    ):
        raise ValueError("steps must be a positive integer and sway finite in [-1,0]")
    if times is None:
        times = torch.linspace(0, 1, steps + 1, device=device)
        times = times + sway * (torch.cos(times * math.pi / 2) - 1 + times)
    else:
        times = torch.as_tensor(times, dtype=torch.float32, device=device)
    if times.shape != (steps + 1,) or not torch.isfinite(times).all():
        raise ValueError("Time grid must be finite [steps+1]")
    if times[0] != 0 or times[-1] != 1 or (times[1:] <= times[:-1]).any():
        raise ValueError("Time grid must start at 0, end at 1 and be strictly increasing")
    return times


@torch.inference_mode()
def sample(
    model,
    prompt,
    prompt_mask,
    valid,
    tokens,
    segments,
    steps=16,
    guidance=1.5,
    seed=0,
    sway=-1.0,
    return_trajectory=False,
    times=None,
    initial_noise=None,
    stats=None,
    condition_cache=None,
):
    if steps < 1 or not -1 <= sway <= 0 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Invalid sampler settings")
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    audio_shapes(prompt, prompt, prompt_mask, valid, prompt.size(-1))
    mask = mask_values(valid, prompt_mask)
    prompt = sanitize(prompt, prompt_mask)
    x = (
        torch.randn(prompt.shape, device=prompt.device, dtype=prompt.dtype, generator=gen)
        if initial_noise is None
        else initial_noise.clone()
    )
    if x.shape != prompt.shape or x.device != prompt.device or not x.is_floating_point():
        raise ValueError("initial_noise must match prompt [B,L,C] and device")
    if not torch.isfinite(sanitize(x, mask)).all() or not torch.isfinite(prompt).all():
        raise ValueError("Reference and target noise must be finite on active frames")
    x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
    times = time_grid(steps, sway, x.device, times)
    cond = (
        model.conditions(prompt, prompt_mask, tokens, segments)
        if condition_cache is None
        else condition_cache
    )
    # Joint dropout's null features are exactly zero; do not redundantly encode the text.
    null = (torch.zeros_like(cond[0]), cond[1], torch.zeros_like(cond[2])) if guidance != 1 else None
    trajectory = [x.clone()] if return_trajectory else None
    for t0, t1 in zip(times[:-1], times[1:]):
        t = t0.expand(x.size(0))
        v = model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond)
        if guidance != 1:
            null_x = x.masked_fill(prompt_mask[..., None], 0)
            u = model(
                null_x,
                t,
                torch.zeros_like(prompt),
                torch.zeros_like(prompt_mask),
                valid,
                tokens,
                segments,
                cached=null,
            )
            v = u + guidance * (v - u)
        x = x + (t1 - t0) * sanitize(v, mask)
        x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
        if trajectory is not None:
            trajectory.append(x.clone())
    if stats is not None:
        stats.update(
            forward_calls=steps * (1 if guidance == 1 else 2),
            branch_evaluations=steps * (1 if guidance == 1 else 2),
            time_grid=times.cpu().tolist(),
            solver="euler",
        )
    return (x, times, torch.stack(trajectory)) if return_trajectory else x
