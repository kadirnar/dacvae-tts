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


# Length-aware RoPE (arXiv:2509.11084): cross-attention positions are gamma * index / length, so a
# frame at 40% of the audio starts out looking at the text around 40% of the transcript.
LENGTH_AWARE_SCALE = 10.0


def rope_angles(positions, head_width):
    """Positions [B,N] on any real scale -> rotation angles [B,1,N,head_width/2]."""
    half = head_width // 2
    frequencies = torch.exp(torch.arange(half, device=positions.device).float() * (-math.log(10000) / half))
    return (positions.float()[..., None] * frequencies)[:, None]


def rotate(x, angles):
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], -1)


class Attention(nn.Module):
    def __init__(self, width, heads, qk_norm=False):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)
        self.q_norm = nn.RMSNorm(width // heads) if qk_norm else None
        self.k_norm = nn.RMSNorm(width // heads) if qk_norm else None

    def forward(self, x, context, valid, query_angles=None, key_angles=None):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if query_angles is not None:
            q, k = rotate(q, query_angles), rotate(k, key_angles)
        y = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=valid[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class TextBlock(nn.Module):
    """Bidirectional self-attention over the transcript: global context the convolutions lack."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = Attention(d, cfg.heads, cfg.qk_norm)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg.ff_mult), nn.GELU(approximate="tanh"), nn.Linear(d * cfg.ff_mult, d)
        )

    def forward(self, x, valid, angles):
        h = self.norm1(x)
        x = x + self.attention(h, h, valid, angles, angles)
        return (x + self.ff(self.norm2(x))) * valid[..., None]


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
        self.blocks = nn.ModuleList([TextBlock(cfg) for _ in range(cfg.text_attention)])
        self.head_width = d // cfg.heads
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, segments):
        valid = tokens.ne(0)
        x = self.embedding(tokens) + self.segment(segments)
        index = torch.arange(tokens.size(1), device=x.device)
        x = x + sinusoidal(index, x.size(-1)).to(x.dtype)
        for conv, mlp in zip(self.convs, self.mlps):
            x = x * valid[..., None]
            x = x + mlp(conv(x.transpose(1, 2)).transpose(1, 2))
        if len(self.blocks):
            angles = rope_angles(index[None], self.head_width)
            x = x * valid[..., None]
            for block in self.blocks:
                x = block(x, valid, angles)
        return self.norm(x) * valid[..., None], valid


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = Attention(d, cfg.heads, cfg.qk_norm)
        self.cross_attn = Attention(d, cfg.heads, cfg.qk_norm)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg.ff_mult), nn.GELU(approximate="tanh"), nn.Linear(d * cfg.ff_mult, d)
        )
        if cfg.adaln_rank:
            # Low-rank per-block correction on top of a modulation shared by every block
            # (PixArt-alpha / EzAudio SOLA / Echo-TTS style); the up projection starts at zero.
            self.ada_down = nn.Linear(d, cfg.adaln_rank)
            self.ada_up = nn.Linear(cfg.adaln_rank, d * 9)
            nn.init.zeros_(self.ada_up.weight)
            nn.init.zeros_(self.ada_up.bias)
        else:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 9))
            nn.init.zeros_(self.ada[-1].weight)
            nn.init.zeros_(self.ada[-1].bias)

    def modulation(self, cond, shared):
        if shared is None:
            return self.ada(cond)
        return shared + self.ada_up(F.silu(self.ada_down(cond)))

    def forward(
        self,
        x,
        text,
        valid,
        text_valid,
        cond,
        self_angles=None,
        query_angles=None,
        key_angles=None,
        shared=None,
    ):
        params = self.modulation(cond, shared).unsqueeze(1).chunk(9, dim=-1)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = params
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.self_attn(h, h, valid, self_angles, self_angles)
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, text, text_valid, query_angles, key_angles)
        x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        return x * valid[..., None]


def _output_dropout(module, args, output):
    return F.dropout(output, module.output_dropout, module.training)


def add_dropout(block, p):
    """Residual-branch dropout where F5-TTS's DiT has it (0.1): after both attention output projections and on
    the FFN hidden activation. Only parameter-free pieces are added (an output hook, a Dropout next to the
    activation), so state-dict keys, initialization and checkpoints are the same with and without it."""
    if not isinstance(block.ff[1], nn.GELU):
        raise TypeError("Dropout expects the block feed-forward as Linear, GELU, Linear")
    block.ff[1] = nn.Sequential(block.ff[1], nn.Dropout(p))
    for attention in (block.self_attn, block.cross_attn):
        attention.output_dropout = p
        attention.register_forward_hook(_output_dropout)


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
        if cfg.dropout:  # at 0 nothing is installed, so default runs draw exactly the same random numbers
            for block in self.blocks:
                add_dropout(block, cfg.dropout)
        self.ada_shared = None
        if cfg.adaln_rank:
            self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(d, d * 9))
            nn.init.zeros_(self.ada_shared[-1].weight)
            nn.init.zeros_(self.ada_shared[-1].bias)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c * p))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        duration_extra = 3 if cfg.duration_features == "text_stats" else 0
        # `rule` derives the length from the prompt's speaking rate instead of a learned head.
        self.duration = (
            nn.Sequential(nn.Linear(2 * d + 1 + duration_extra, d), nn.SiLU(), nn.Linear(d, 1))
            if cfg.duration == "head"
            else None
        )
        # Auxiliary CTC head on intermediate frames (A-DMA, arXiv:2505.19595): training only. It makes
        # the generator route every transcript byte to its frames early, i.e. learn the alignment.
        self.ctc = nn.Linear(d, VOCAB_SIZE) if cfg.ctc_layer else None
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
        if self.duration is None:
            raise ValueError("This model has no duration head; its length follows the prompt speaking rate")
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

    def forward(
        self, x, time, prompt, prompt_mask, valid, tokens, segments, drop=None, cached=None, return_ctc=False
    ):
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
        index = torch.arange(h.size(1), device=x.device)
        angles = (None, None, None)
        if self.cfg.positions == "rope":
            head_width = self.cfg.width // self.cfg.heads
            audio_length = packed_valid.sum(1, keepdim=True).clamp_min(1)
            text_length = text_valid.sum(1, keepdim=True).clamp_min(1)
            text_index = torch.arange(text.size(1), device=x.device)
            angles = (
                rope_angles(index[None], head_width),
                rope_angles(LENGTH_AWARE_SCALE * index[None] / audio_length, head_width),
                rope_angles(LENGTH_AWARE_SCALE * text_index[None] / text_length, head_width),
            )
        else:
            h = h + sinusoidal(index, h.size(-1)).to(h.dtype)
        time_embedding = self.time(sinusoidal(time * 1000, self.cfg.width).to(h.dtype))
        if time_embedding.shape != voice.shape:
            raise ValueError("Time embedding and reference summary must both be [B,D]")
        cond = time_embedding + voice
        shared = self.ada_shared(cond) if self.ada_shared is not None else None
        ctc_logits = None
        for number, block in enumerate(self.blocks, 1):
            args = (h, text, packed_valid, text_valid, cond, *angles, shared)
            h = (
                checkpoint(block, *args, use_reentrant=False)
                if self.grad_checkpoint and self.training
                else block(*args)
            )
            if return_ctc and number == self.cfg.ctc_layer:
                ctc_logits = self.ctc(h)
        output = self.output(h)
        expected_packs = (length + pad) // p
        if output.shape != (b, expected_packs, channels * p):
            raise ValueError("Velocity projection returned an unexpected packed length or width")
        # Remove only the explicitly added packing padding, then mask at frame resolution.
        velocity = sanitize(output.reshape(b, length + pad, channels)[:, :length], valid)
        return (velocity, ctc_logits, packed_valid) if return_ctc else velocity


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


def prediction_kind(model):
    return getattr(getattr(model, "cfg", None), "prediction", "velocity")


def to_velocity(model, output, x, time):
    """EDM-style preconditioning (unit-variance data, linear path): the network output F has unit
    variance at every t, x1_hat = c_skip x + c_out F, and the 1/(1-t) factor cancels analytically."""
    if prediction_kind(model) != "edm":
        return output
    t = time[:, None, None].to(x.dtype)
    scale = t.square() + (1 - t).square()
    return ((2 * t - 1) / scale) * x + output / scale.sqrt()


def sample_time(count, device, mode="uniform"):
    if mode == "uniform":
        return torch.rand(count, device=device)
    if mode != "logit_normal":
        raise ValueError("time sampling must be uniform or logit_normal")
    # Stratified logit-normal(0,1): one draw per equal-probability slice of the batch.
    uniform = (torch.randperm(count, device=device) + torch.rand(count, device=device)) / count
    return torch.sigmoid(torch.special.ndtri(uniform.clamp(1e-4, 1 - 1e-4))).clamp(1e-3, 1 - 1e-3)


def flow_loss(
    model,
    batch,
    dropout=0.1,
    time=None,
    noise=None,
    return_details=False,
    cached=None,
    time_sampling="uniform",
    guidance_weight=0.0,
):
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must lie in [0,1]")
    mask = mask_values(batch["valid"], batch["prompt_mask"])
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    time = sample_time(b, x1.device, time_sampling) if time is None else time
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
    with_ctc = return_details and getattr(model, "ctc", None) is not None and model.training
    pred = model(
        xt,
        time,
        batch["prompt"],
        batch["prompt_mask"],
        batch["valid"],
        batch["tokens"],
        batch["segments"],
        drop=drop,
        **({} if cached is None else {"cached": cached}),
        **({"return_ctc": True} if with_ctc else {}),
    )
    ctc = None
    if with_ctc:
        pred, logits, token_valid = pred
        ctc = ctc_alignment_loss(logits, token_valid, batch["tokens"], drop)
    target = x1 - noise
    if prediction_kind(model) == "edm":
        t = time[:, None, None]
        target = ((1 - t) * x1 - t * noise) / (t.square() + (1 - t).square()).sqrt()
    if guidance_weight:
        target = target + guidance_weight * guidance_direction(model, pred, xt, time, batch, drop, cached)
    losses = per_example_mse(pred, target, mask)
    if return_details:
        counts = mask.sum(1)
        rms = (sanitize(pred.float(), mask).square().sum((1, 2)) / (counts * pred.size(-1))).sqrt()
        details = {
            "flow": losses,
            "times": time,
            "frames": counts,
            "prediction_rms": rms,
            "noise": noise,
            "drop": drop,
        }
        if ctc is not None:
            details["ctc"] = ctc
        return details
    return losses


def guidance_direction(model, prediction, xt, time, batch, drop, cached=None):
    """sg(out_cond - out_null) at the training (x_t, t), the model-guidance direction; 0 on CFG-dropped rows.

    The null branch drops text, voice and prompt exactly like condition dropout (and the sampler's CFG null
    branch). Output differences are the right quantity in either parameterization: EDM's `to_velocity` is
    affine in the output with slope 1/sqrt(t^2 + (1-t)^2) and an offset that depends only on (x_t, t), the
    same for both branches on target frames, so target + w (F_cond - F_null) is exactly the F-space image of
    the velocity target v + w (v_cond - v_null). Without model dropout the conditional output is the training
    forward itself (same inputs and weights); with dropout both branches are recomputed in eval mode so the
    direction carries no dropout noise.
    """
    inputs = (xt, time, batch["prompt"], batch["prompt_mask"], batch["valid"])
    inputs += (batch["tokens"], batch["segments"])
    extra = {} if cached is None else {"cached": cached}
    with torch.no_grad():
        if model.training and getattr(model.cfg, "dropout", 0) > 0:
            model.eval()
            try:
                cond = model(*inputs, drop=torch.zeros_like(drop), **extra)
                null = model(*inputs, drop=torch.ones_like(drop), **extra)
            finally:
                model.train()
        else:
            cond = prediction.detach()
            null = model(*inputs, drop=torch.ones_like(drop), **extra)
    return (cond - null).masked_fill(drop[:, None, None], 0)


def ctc_alignment_loss(logits, token_valid, tokens, drop):
    """Per-example CTC between generator frames and transcript bytes; PAD (0) is the blank.

    Examples whose text was dropped for classifier-free guidance cannot be aligned and get zero.
    """
    from .text import BYTE_OFFSET

    targets = [row[row >= BYTE_OFFSET] for row in tokens]
    lengths = torch.tensor([len(row) for row in targets], device=logits.device)
    loss = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        torch.cat(targets),
        token_valid.sum(1),
        lengths,
        blank=0,
        reduction="none",
        zero_infinity=True,
    )
    return (loss / lengths.clamp_min(1)).masked_fill(drop, 0)


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


def _masked_stats(x, mask):
    """Per-example (sum of squares, element count) over active frames of [B,L,C]."""
    x = sanitize(x.float(), mask)
    return x.square().sum((1, 2)), (mask.sum(1) * x.size(-1)).clamp_min(1)


def guided_update(v_cond, v_null, x, t, mask, guidance, rescale=0.0, eta=1.0, norm=0.0, momentum=0.0, state=None):
    """Guided velocity from the conditional and null velocities.

    Guidance acts on the clean-data estimate x1 = x + (1 - t) v of the linear path. With the defaults this is exactly
    classifier-free guidance, v_null + g (v_cond - v_null). APG (Sadat et al., adaptive projected guidance) splits the
    difference x1_cond - x1_null into its component parallel to x1_cond, which mostly raises amplitude and causes
    over-saturation at high guidance, and the orthogonal rest; the parallel part is weighted by `eta`. `momentum` < 0
    is APG's reverse momentum (running average kept in `state`), `norm` > 0 caps the per-element RMS of the
    difference. `rescale` (phi of Lin et al.) blends the guided x1 with a copy rescaled to the conditional x1's
    per-utterance standard deviation. All statistics are taken over the target frames of each example.
    """
    if rescale == 0 and eta == 1 and norm == 0 and momentum == 0:
        return v_null + guidance * (v_cond - v_null)
    one_minus_t = (1 - t)[:, None, None].to(x.dtype)
    x1_cond = x + one_minus_t * v_cond
    diff = sanitize(one_minus_t * (v_cond - v_null), mask)
    if momentum:
        running = state.get("running")
        diff = diff if running is None else diff + momentum * running
        state["running"] = diff
    if norm > 0:
        squares, count = _masked_stats(diff, mask)
        rms = (squares / count).sqrt()
        diff = diff * (norm / rms.clamp_min(1e-8)).clamp(max=1.0)[:, None, None].to(diff.dtype)
    if eta != 1:
        reference = sanitize(x1_cond.float(), mask)
        unit = reference / reference.square().sum((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        parallel = (diff.float() * unit).sum((1, 2), keepdim=True) * unit
        diff = (diff.float() - parallel + eta * parallel).to(diff.dtype)
    x1 = x1_cond + (guidance - 1) * diff
    if rescale > 0:
        def centered_std(value):
            value = sanitize(value.float(), mask)
            count = (mask.sum(1) * value.size(-1)).clamp_min(1)
            mean = value.sum((1, 2)) / count
            var = sanitize((value - mean[:, None, None]).square(), mask).sum((1, 2)) / count
            return var.clamp_min(1e-12).sqrt()

        factor = (centered_std(x1_cond) / centered_std(x1))[:, None, None].to(x1.dtype)
        x1 = rescale * (x1 * factor) + (1 - rescale) * x1
    return (x1 - x) / one_minus_t


def text_only_rows(model, full_valid, prompt_mask, tokens, segments):
    """Inputs of the prompt-free branch of independent guidance: each example's target frames as their own sequence.

    Training drops the voice prompt entirely (prompt dropout: no reference frames, transcript of the target only),
    so the text-only branch is exactly that condition. `tokens`/`segments` hold the target text alone per example.
    Returns a dict with the gather index [B,T] into the full sequence, valid [B,T] and cached conditions.
    """
    target = full_valid & ~prompt_mask
    lengths = target.sum(1)
    width = int(lengths.max())
    positions = torch.arange(width, device=full_valid.device)[None]
    valid = positions < lengths[:, None]
    start = prompt_mask.sum(1, keepdim=True)
    index = torch.where(valid, start + positions, torch.zeros_like(positions)).long()
    empty = torch.zeros(len(lengths), width, model.cfg.latent_dim, device=full_valid.device)
    no_prompt = torch.zeros_like(valid)
    return dict(
        index=index,
        valid=valid,
        prompt=empty,
        prompt_mask=no_prompt,
        tokens=tokens,
        segments=segments,
        cached=model.conditions(empty, no_prompt, tokens, segments),
    )


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
    guidance_until=1.0,
    noise_scale=1.0,
    guidance_from=0.0,
    cfg_rescale=0.0,
    apg_eta=1.0,
    apg_norm=0.0,
    apg_momentum=0.0,
    speaker_guidance=None,
    text_only=None,
):
    """Euler sampler with classifier-free guidance.

    Guidance is applied while guidance_from <= t < guidance_until (t=0 is noise): Echo/Irodori guide only the noisy
    half of the trajectory (cfg_min_t 0.5 in their reversed convention), which keeps the alignment benefit at roughly
    half the extra forward passes. `noise_scale` shrinks the initial noise (Echo's 0.8-0.9 "truncation").
    `cfg_rescale`, `apg_eta`, `apg_norm` and `apg_momentum` reshape the guided update (see `guided_update`); their
    defaults give plain CFG. `speaker_guidance` enables independent text/speaker guidance with a third, prompt-free
    branch built by `text_only_rows` (pass it as `text_only`): v_null + g (v_text - v_null) + g_s (v_full - v_text),
    which equals plain CFG for g_s = g.
    """
    if steps < 1 or not -1 <= sway <= 0 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Invalid sampler settings")
    if not 0 < guidance_until <= 1 or not 0 < noise_scale <= 1.5 or not 0 <= guidance_from < guidance_until:
        raise ValueError("Need 0 <= guidance_from < guidance_until <= 1 and noise_scale in (0,1.5]")
    if not 0 <= cfg_rescale <= 1 or apg_norm < 0 or not -1 < apg_momentum < 1 or not math.isfinite(apg_eta):
        raise ValueError("cfg_rescale must lie in [0,1], apg_norm >= 0, apg_momentum in (-1,1)")
    if speaker_guidance is not None and (text_only is None or not math.isfinite(speaker_guidance)):
        raise ValueError("Independent speaker guidance needs the prompt-free branch (text_only_rows)")
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    audio_shapes(prompt, prompt, prompt_mask, valid, prompt.size(-1))
    mask = mask_values(valid, prompt_mask)
    prompt = sanitize(prompt, prompt_mask)
    x = (
        torch.randn(prompt.shape, device=prompt.device, dtype=prompt.dtype, generator=gen) * noise_scale
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
    if guidance != 1:
        # Conditioned and null branches share one forward pass. Joint dropout's null features are
        # exactly zero, so the text is not encoded a second time.
        zeros = torch.zeros_like
        pair = dict(
            prompt=torch.cat([prompt, zeros(prompt)]),
            prompt_mask=torch.cat([prompt_mask, zeros(prompt_mask)]),
            valid=valid.repeat(2, 1),
            tokens=tokens.repeat(2, 1),
            segments=segments.repeat(2, 1),
            cached=(
                torch.cat([cond[0], zeros(cond[0])]),
                cond[1].repeat(2, 1),
                torch.cat([cond[2], zeros(cond[2])]),
            ),
        )
    if speaker_guidance is not None:
        index = text_only["index"][..., None].expand(-1, -1, x.size(-1))
        text_valid = text_only["valid"]
        branch = {k: text_only[k] for k in ("prompt", "prompt_mask", "tokens", "segments", "cached")}
    trajectory = [x.clone()] if return_trajectory else None
    guided_steps = evaluations = 0
    apg_state = {}
    for t0, t1 in zip(times[:-1], times[1:]):
        t = t0.expand(x.size(0))
        if guidance == 1 or not guidance_from <= float(t0) < guidance_until:
            v = to_velocity(
                model, model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond), x, t
            )
            evaluations += 1
        else:
            both = torch.cat([x, x.masked_fill(prompt_mask[..., None], 0)])
            v, u = to_velocity(model, model(both, t.repeat(2), **pair), both, t.repeat(2)).chunk(2)
            evaluations += 2
            if speaker_guidance is None:
                v = guided_update(v, u, x, t, mask, guidance, cfg_rescale, apg_eta, apg_norm, apg_momentum, apg_state)
            else:
                rows = sanitize(torch.gather(x, 1, index), text_valid)
                w = to_velocity(model, model(rows, t, valid=text_valid, **branch), rows, t)
                w = torch.zeros_like(x).scatter_add(1, index, sanitize(w, text_valid))
                v = u + guidance * (w - u) + speaker_guidance * (v - w)
                evaluations += 1
            guided_steps += 1
        x = x + (t1 - t0) * sanitize(v, mask)
        x = sanitize(torch.where(prompt_mask[..., None], prompt, x), valid)
        if trajectory is not None:
            trajectory.append(x.clone())
    if stats is not None:
        stats.update(
            forward_calls=steps + (guided_steps if speaker_guidance is not None else 0),
            branch_evaluations=evaluations,
            guidance_until=guidance_until,
            guidance_from=guidance_from,
            noise_scale=noise_scale,
            cfg_rescale=cfg_rescale,
            apg=dict(eta=apg_eta, norm=apg_norm, momentum=apg_momentum),
            speaker_guidance=speaker_guidance,
            time_grid=times.cpu().tolist(),
            solver="euler",
        )
    return (x, times, torch.stack(trajectory)) if return_trajectory else x
