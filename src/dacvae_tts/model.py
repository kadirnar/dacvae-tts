"""Reference-conditioned, non-autoregressive continuous latent flow model."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import RepaProjector, SpeakerAlignment
from .config import ModelConfig
from .contracts import audio_shapes, mask_values, sanitize, text_shapes
from .reference import ReferencePool, TemporalReference
from .speed import block_checkpoint, run_block
from .text import BYTE_OFFSET, CHAR_VOCAB_SIZE, CONTINUATION, VOCAB_SIZE, char_ctc_targets


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
    def __init__(self, width, heads, qk_norm=False, gate=False):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)
        self.q_norm = nn.RMSNorm(width // heads) if qk_norm else None
        self.k_norm = nn.RMSNorm(width // heads) if qk_norm else None
        # Head-wise output gate y_h <- 2 sigmoid(w_h . x + b_h) y_h from the query-side input (Qwen gated
        # attention, arXiv:2505.06708: query-dependent sparsity, no attention sink, higher-LR stability; Echo,
        # Irodori and Darya gate too; no TTS ablation). 2 sigmoid with zero init is exactly 1: starts as the
        # baseline and can still open to 2. skip_init draws no random numbers (baseline weights per seed).
        self.gate = None
        if gate:
            self.gate = nn.utils.skip_init(nn.Linear, width, heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self, x, context, valid, query_angles=None, key_angles=None, value_mix=None, first_value=None
    ):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        if value_mix is not None:
            # Value residual: v <- l1 v + l2 v_1 with the first block's raw values [B,H,N,D/H], which are
            # returned for the later blocks (the first block has none yet and mixes its own).
            first_value = v if first_value is None else first_value
            v = value_mix[0] * v + value_mix[1] * first_value
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if query_angles is not None:
            q, k = rotate(q, query_angles), rotate(k, key_angles)
        y = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=valid[:, None, None, :])
        if self.gate is not None:
            y = y * (2 * torch.sigmoid(self.gate(x))).transpose(1, 2)[..., None].to(y.dtype)
        y = self.out(y.transpose(1, 2).reshape(b, n, d))
        return y if value_mix is None else (y, first_value)


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


class SpeakerContext(nn.Module):
    """Multi-clip speaker context -> one vector [B,D] for the voice condition (model.speaker_context: vector).

    Latents [B,L,C] (the clips back to back, masked) are patched `speaker_context_patch` frames per token, projected
    to `speaker_context_width`, mixed by `speaker_context_layers` bidirectional self-attention blocks without
    positions (a set of patches: the clip order does not matter), attention-pooled and normalized; the zero-init
    output projection starts the model as the baseline. Rows without context contribute exactly zero.
    """

    def __init__(self, cfg):
        super().__init__()
        width, self.patch = cfg.speaker_context_width, cfg.speaker_context_patch
        inner = type("ContextConfig", (), dict(width=width, heads=cfg.speaker_context_heads, qk_norm=cfg.qk_norm,
                                               ff_mult=cfg.ff_mult))
        self.input = nn.Linear(cfg.latent_dim * self.patch, width)
        self.blocks = nn.ModuleList([TextBlock(inner) for _ in range(cfg.speaker_context_layers)])
        self.pool = ReferencePool(width, "attention")
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, cfg.width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context, mask):
        b, length, channels = context.shape
        pad = (-length) % self.patch
        context = F.pad(sanitize(context, mask), (0, 0, 0, pad))
        valid = F.pad(mask, (0, pad)).reshape(b, -1, self.patch).all(-1)  # a token needs all its frames
        present = valid.any(1)
        attend = valid.clone()
        attend[:, 0] |= ~present  # empty rows attend to one token (no NaN); their result is zeroed below
        x = self.input(context.reshape(b, -1, channels * self.patch)) * attend[..., None]
        for block in self.blocks:
            x = block(x, attend, None)
        pooled = self.pool(x, valid)
        return self.output(self.norm(pooled)) * present[:, None].to(pooled.dtype)


class SwiGLU(nn.Module):
    """silu(gate) * value from one fused projection; its two row halves are separate maps for Muon."""

    def __init__(self, width, hidden):
        super().__init__()
        self.proj = nn.Linear(width, 2 * hidden)

    def forward(self, x):
        gate, value = self.proj(x).chunk(2, dim=-1)
        return F.silu(gate) * value


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = Attention(d, cfg.heads, cfg.qk_norm, cfg.attn_gate == "head")
        self.cross_attn = Attention(d, cfg.heads, cfg.qk_norm, cfg.attn_gate == "head")
        if cfg.ffn_activation == "swiglu":
            # Equal parameters: hidden 2/3 of the GELU width, rounded to a multiple of 64 (1024 at 512 x 3).
            # T5 and LightningDiT gain from GLUs; the 140M SR-DiT ablation was neutral: an A/B, not a default.
            hidden = max(64, 64 * round(2 * d * cfg.ff_mult / 3 / 64))
            self.ff = nn.Sequential(SwiGLU(d, hidden), nn.Linear(hidden, d))
        else:
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
        # Value residual (ResFormer, arXiv:2410.17897; 140M SR-DiT FID 4.02 -> 3.64): self-attention values
        # v_l <- l1 v_l + l2 v_1 keep the first block's token features reachable in deep blocks. Two scalars
        # per block, identity at init (l1 = 1, l2 = 0). Block 1 mixes its own values: all are used (DDP).
        self.value_mix = nn.Parameter(torch.tensor([1.0, 0.0])) if cfg.value_residual else None
        # Depthwise time convolution on the FFN hidden units, after the activation and residual (Mix-FFN
        # style): the only local mixing along frames in the generator. ZipVoice WER 1.69 -> 9.79 without
        # its conv modules, FastSpeech CMOS -0.11 without conv in the FFN, U-DiT/SANA gains; F5's
        # Conv2Audio is the counter-example (4.17 -> 5.78). Zero-init, so the model starts as the baseline;
        # skip_init draws no random numbers, so the other weights stay the baseline's for the same seed.
        self.ff_conv = None
        if cfg.ffn_conv_kernel:
            hidden, kernel = self.ff[-1].in_features, cfg.ffn_conv_kernel
            self.ff_conv = nn.utils.skip_init(
                nn.Conv1d, hidden, hidden, kernel, padding=kernel // 2, groups=hidden
            )
            nn.init.zeros_(self.ff_conv.weight)
            nn.init.zeros_(self.ff_conv.bias)

    def modulation(self, cond, shared):
        if shared is None:
            return self.ada(cond)
        return shared + self.ada_up(F.silu(self.ada_down(cond)))

    def conv_feed_forward(self, h, valid):
        """The FFN with its depthwise time convolution between the activation and the output projection."""
        *project, down = self.ff
        for layer in project:
            h = layer(h)
        h = h * valid[..., None]  # padded frames hold bias-driven activations: keep them from neighbours
        return down(h + self.ff_conv(h.transpose(1, 2)).transpose(1, 2))

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
        first_value=None,
    ):
        """Returns x, or (x, first-block values) with value_residual; the values are passed explicitly so
        activation checkpointing recomputes each block from its inputs alone."""
        params = self.modulation(cond, shared).unsqueeze(1).chunk(9, dim=-1)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = params
        h = self.norm1(x) * (1 + s1) + b1
        if self.value_mix is None:
            x = x + g1 * self.self_attn(h, h, valid, self_angles, self_angles)
        else:
            mix = self.value_mix
            y, first_value = self.self_attn(h, h, valid, self_angles, self_angles, mix, first_value)
            x = x + g1 * y
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, text, text_valid, query_angles, key_angles)
        if self.ff_conv is None:
            x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        else:
            x = x + g3 * self.conv_feed_forward(self.norm3(x) * (1 + s3) + b3, valid)
        x = x * valid[..., None]
        return x if self.value_mix is None else (x, first_value)


def _output_dropout(module, args, output):
    if isinstance(output, tuple):  # value-residual self-attention (#9): (output, first-block values)
        return (F.dropout(output[0], module.output_dropout, module.training), *output[1:])
    return F.dropout(output, module.output_dropout, module.training)


def add_dropout(block, p):
    """Residual-branch dropout where F5-TTS's DiT has it (0.1): after both attention output projections and on
    the FFN hidden activation. Only parameter-free pieces are added (an output hook, a Dropout next to the
    activation), so state-dict keys, initialization and checkpoints are the same with and without it.

    The #9 block options keep the same placement: the GELU FFN (also with ffn_conv_kernel, whose depthwise
    convolution then sees the dropped activation) gets the Dropout next to its GELU; the SwiGLU FFN
    ([SwiGLU, Linear]) gets the output hook on its gated activation instead, which keeps its state-dict keys;
    the attention hook drops only the attention output of a value-residual self-attention, whose first-block
    values pass through unchanged, and acts after the head gate (attn_gate), i.e. on what the block adds.
    """
    if isinstance(block.ff[0], SwiGLU):
        block.ff[0].output_dropout = p
        block.ff[0].register_forward_hook(_output_dropout)
    elif len(block.ff) == 3 and isinstance(block.ff[1], nn.GELU):
        block.ff[1] = nn.Sequential(block.ff[1], nn.Dropout(p))
    else:
        raise TypeError("Dropout expects the block feed-forward as Linear, GELU, Linear or SwiGLU, Linear")
    for attention in (block.self_attn, block.cross_attn):
        attention.output_dropout = p
        attention.register_forward_hook(_output_dropout)


def set_dropout(model, p):
    """Set the rate of the residual-branch dropout that `add_dropout` installed in the generator blocks; 0
    turns it off in training mode too (e.g. for GRPO, whose PPO ratio needs the rollout policy)."""
    for block in model.blocks:
        for module in block.modules():
            if hasattr(module, "output_dropout"):
                module.output_dropout = p
            elif isinstance(module, nn.Dropout):
                module.p = p


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
        # the generator route every transcript byte (or letter, ctc_targets: chars) to its frames early,
        # i.e. learn the alignment.
        labels = CHAR_VOCAB_SIZE if cfg.ctc_targets == "chars" else VOCAB_SIZE
        self.ctc = nn.Linear(d, labels) if cfg.ctc_layer else None
        # Input -> output long skip: the input embedding re-enters just before the output head, fused with the
        # last block by LN + Linear over [h_0, h_L] (the pre-norm matches their scales; the head's LayerNorm
        # then normalizes the sum, Hunyuan-DiT's fix for loss spikes after skip fusion). DiTTo, a
        # cross-attention DiT like this one: WER 3.30 -> 2.93, SIM 0.573 -> 0.588; EzAudio: faster
        # convergence. Counter-evidence is in-context only (F5 4.17 -> 5.17). Zero-init: starts as baseline.
        self.skip = None
        if cfg.long_skip:
            self.skip = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d))
            nn.init.zeros_(self.skip[-1].weight)
            nn.init.zeros_(self.skip[-1].bias)
        # Final adaLN (the DiT / F5 final layer): shift and scale of the output LayerNorm come from the
        # condition, so the velocity head can adapt to flow time and voice. Rank-r like the blocks when
        # adaln_rank > 0, else a full D -> 2D map; the last projection starts at zero (baseline at init).
        self.final_ada = None
        if cfg.final_adaln:
            r = cfg.adaln_rank
            self.final_ada = (
                nn.Sequential(nn.Linear(d, r), nn.SiLU(), nn.Linear(r, 2 * d))
                if r
                else nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
            )
            nn.init.zeros_(self.final_ada[-1].weight)
            nn.init.zeros_(self.final_ada[-1].bias)
        # Pooled transcript in the condition (DiTTo: WER 3.00 -> 2.93): the masked mean of the byte encodings
        # of segment 1 joins time + voice, so every adaLN sees the whole sentence. Segment 1 is the target text
        # in the `segments` layout but the whole stream, prompt transcript + target, in `joined` (every Turkish
        # config). No bias and zero-init: it starts as the baseline, and text dropped for guidance (all zeros)
        # adds nothing to the null branch.
        self.text_pool = None
        if cfg.cond_text_pool:
            self.text_pool = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.text_pool.weight)
        # Frozen speaker-verification embedding -> voice condition (bias-free and zero-init: starts as the baseline,
        # and a zero row, i.e. no prompt, adds exactly nothing). skip_init draws no random numbers.
        self.speaker_condition = None
        if cfg.speaker_condition_dim:
            self.speaker_condition = nn.utils.skip_init(nn.Linear, cfg.speaker_condition_dim, d, bias=False)
            nn.init.zeros_(self.speaker_condition.weight)
        # Training-only teacher heads (alignment.py): created last and only when configured, so models
        # without them keep their initialization and old checkpoints load strictly.
        self.repa = RepaProjector(d, cfg.repa_dim, p) if cfg.repa_layer else None
        self.tla = (
            SpeakerAlignment(d, cfg.tla_dim, len(cfg.tla_layers), cfg.tla_hidden) if cfg.tla_layers else None
        )
        # Multi-clip speaker context: created last, so every other weight equals the baseline's for a seed.
        self.speaker_context = SpeakerContext(cfg) if cfg.speaker_context == "vector" else None
        self.grad_checkpoint = False
        self.strict_checks = True  # train.strict_checks: false skips value checks that wait for the GPU
        self.block_runner = run_block  # train.compile: blocks swaps in a compiled runner (speed.py)

    def reference_summary(self, prompt, prompt_mask, speaker=None, context=None, context_mask=None):
        """Global voice vector [B,D] of the prompt; plus the projected speaker embedding `speaker` [B,E] for models
        with `speaker_condition_dim` (rows without a prompt or with a zero embedding get none), and the speaker
        context vector of `context` [B,L,C] / `context_mask` [B,L] for models with `speaker_context` (None: none)."""
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
        if self.speaker_condition is not None:
            width = self.cfg.speaker_condition_dim
            if speaker is None or speaker.shape != (prompt.size(0), width):
                raise ValueError(f"This model is conditioned on a speaker embedding: pass speaker [B,{width}] "
                                 "(zeros where there is none)")
            present = (prompt_mask.any(1) & speaker.ne(0).any(-1))[:, None]
            unit = F.normalize(speaker.float(), dim=-1) * width**0.5 * present  # unit-variance entries
            voice = voice + self.speaker_condition(unit.to(voice.dtype))
        if self.speaker_context is not None and context is not None:
            if context.ndim != 3 or context.shape[0] != prompt.size(0) or context_mask.shape != context.shape[:2]:
                raise ValueError("Speaker context must be [B,L,C] latents with a [B,L] mask")
            voice = voice + self.speaker_context(context, context_mask)
        elif context is not None:
            raise ValueError("This model has no speaker context (model.speaker_context: none)")
        return voice

    def conditions(self, prompt, prompt_mask, tokens, segments, drop=None, speaker=None, context=None,
                   context_mask=None):
        if self.strict_checks and self.cfg.text_units == "chars":
            low, high = CONTINUATION  # UTF-8 continuation bytes never occur in character-unit rows
            if ((tokens >= low) & (tokens <= high)).any():
                raise ValueError("UTF-8 byte ids given to a character-unit model (text.to_units converts them)")
        text, text_valid = self.text(tokens, segments)
        voice = self.reference_summary(prompt, prompt_mask, speaker, context, context_mask)
        if drop is not None:
            text = text.masked_fill(drop[:, None, None], 0)
            voice = voice.masked_fill(drop[:, None], 0)
        return text, text_valid, voice

    def predict_duration(self, prompt, prompt_mask, tokens, segments, cached=None, speaker=None):
        if self.duration is None:
            raise ValueError("This model has no duration head; its length follows the prompt speaking rate")
        if (
            prompt.ndim != 3
            or prompt.shape[-1] != self.cfg.latent_dim
            or prompt_mask.shape != prompt.shape[:2]
        ):
            raise ValueError("Duration prompt must be [B,L,C] with matching [B,L] reference mask")
        text_shapes(tokens, segments, prompt.size(0), prompt.device)
        text, _, voice = (
            self.conditions(prompt, prompt_mask, tokens, segments, speaker=speaker) if cached is None else cached
        )
        target_bytes = (segments == 1) & (tokens >= BYTE_OFFSET)
        if self.strict_checks and (target_bytes.sum(1) == 0).any():
            raise ValueError("Duration prediction requires nonempty target text")
        ref_bytes = ((segments == 0) & (tokens >= BYTE_OFFSET)).sum(1).clamp_min(1)
        rate = (prompt_mask.sum(1).float().clamp_min(1) / ref_bytes).log().unsqueeze(1)
        features = [masked_mean(text, target_bytes), voice, rate]
        if self.cfg.duration_features == "text_stats":
            byte_values = tokens - BYTE_OFFSET
            # Bytes that start a character; character units (Latin-5) never take continuation values, so this
            # counts characters for both text units.
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
        self,
        x,
        time,
        prompt,
        prompt_mask,
        valid,
        tokens,
        segments,
        drop=None,
        cached=None,
        return_ctc=False,
        return_hidden=(),
        speaker=None,
        context=None,
        context_mask=None,
    ):
        """Velocity [B,L,C]; with `return_ctc`, (velocity, CTC logits, packed mask). A nonempty
        `return_hidden` (1-based block indices) wraps that result as (result, {block: [B,N,D]}) for
        training-only objectives that read intermediate states."""
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
            cached = self.conditions(prompt, prompt_mask, tokens, segments, drop, speaker, context, context_mask)
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
        if self.text_pool is not None:  # segment-1 bytes (joined: all of them); `text` is zero where dropped
            cond = cond + self.text_pool(masked_mean(text, (segments == 1) & (tokens >= BYTE_OFFSET)))
        shared = self.ada_shared(cond) if self.ada_shared is not None else None
        ctc_logits = None
        first = h  # input embedding h_0, for the optional long skip
        first_value = None  # first block's self-attention values, for the optional value residual
        hidden = {}
        for number, block in enumerate(self.blocks, 1):
            args = (h, text, packed_valid, text_valid, cond, *angles, shared)
            if self.cfg.value_residual:
                args = (*args, first_value)
            # block_runner/block_checkpoint (speed.py): every train.grad_checkpoint mode and compile: blocks
            # see the value-residual input and (x, first-block values) output like any other block tensor.
            h = self.block_runner(block, args, block_checkpoint(self.grad_checkpoint, number, self.training))
            if self.cfg.value_residual:
                h, first_value = h
            if return_ctc and number == self.cfg.ctc_layer:
                ctc_logits = self.ctc(h)
            if number in return_hidden:
                hidden[number] = h
        if self.skip is not None:
            h = h + self.skip(torch.cat([first, h], -1))
        if self.final_ada is None:
            output = self.output(h)
        else:
            shift, scale = self.final_ada(cond).unsqueeze(1).chunk(2, dim=-1)
            output = self.output[1](self.output[0](h) * (1 + scale) + shift)
        expected_packs = (length + pad) // p
        if output.shape != (b, expected_packs, channels * p):
            raise ValueError("Velocity projection returned an unexpected packed length or width")
        # Remove only the explicitly added packing padding, then mask at frame resolution.
        velocity = sanitize(output.reshape(b, length + pad, channels)[:, :length], valid)
        result = (velocity, ctc_logits, packed_valid) if return_ctc else velocity
        return (result, hidden) if return_hidden else result


def per_example_mse(prediction, target, mask, strict=True):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or mask.shape != prediction.shape[:2]
        or mask.dtype != torch.bool
    ):
        raise ValueError("MSE requires matching [B,L,C] predictions/targets and boolean [B,L] mask")
    if strict and (mask.sum(-1) == 0).any():
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


def flow_target(model, x1, noise, time):
    """Regression target of the linear path x_t=(1-t)noise+t x1 in the model's parameterization.

    Velocity: x1 - noise. EDM: the unit-variance F that `to_velocity` inverts. Latent negatives
    (ΔFM) evaluate it on corrupted x1 with the positive's noise and time, so both must agree.
    """
    target = x1 - noise
    if prediction_kind(model) == "edm":
        t = time[:, None, None]
        target = ((1 - t) * x1 - t * noise) / (t.square() + (1 - t).square()).sqrt()
    return target


def flow_loss(
    model,
    batch,
    dropout=0.1,
    time=None,
    noise=None,
    return_details=False,
    cached=None,
    time_sampling="uniform",
    hidden_layers=(),
    guidance_weight=0.0,
):
    """Per-example flow losses; `return_details` adds diagnostics, the CTC term and, for nonempty
    `hidden_layers`, the requested block outputs under "hidden" (training-only teacher terms)."""
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must lie in [0,1]")
    strict = getattr(model, "strict_checks", True)
    mask = mask_values(batch["valid"], batch["prompt_mask"], strict=strict)
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    time = sample_time(b, x1.device, time_sampling) if time is None else time
    noise = torch.randn_like(x1) if noise is None else noise
    if time.shape != (b,) or noise.shape != x1.shape or time.device != x1.device or noise.device != x1.device:
        raise ValueError("Noise must match [B,L,C]; time must match [B], on the audio device")
    if strict and (not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any()):
        raise ValueError("Flow time must be finite and lie in [0,1]")
    noise = sanitize(noise, batch["valid"])
    xt = (1 - time[:, None, None]) * noise + time[:, None, None] * x1
    # The conditioning prefix follows exactly the same path at training and inference.
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    drop = torch.rand(b, device=x1.device) < dropout
    # Remove reference from the state too, otherwise CFG's null branch leaks the voice.
    xt = xt.masked_fill((drop[:, None] & batch["prompt_mask"])[..., None], 0)
    with_ctc = return_details and getattr(model, "ctc", None) is not None and model.training
    with_hidden = return_details and bool(hidden_layers)
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
        **(condition_inputs(batch) if cached is None else {}),
        **({"return_ctc": True} if with_ctc else {}),
        **({"return_hidden": tuple(hidden_layers)} if with_hidden else {}),
    )
    hidden = None
    if with_hidden:
        pred, hidden = pred
    ctc = None
    if with_ctc:
        pred, logits, token_valid = pred
        ctc = ctc_alignment_loss(logits, token_valid, batch["tokens"], drop, *ctc_labels(model, batch))
    target = flow_target(model, x1, noise, time)
    offset = None
    if guidance_weight:
        offset = guidance_weight * guidance_direction(model, pred, xt, time, batch, drop, cached)
        target = target + offset
    losses = per_example_mse(pred, target, mask, strict)
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
            "prediction": pred,  # lets target-only terms (latent negatives) reuse this generator pass
        }
        if ctc is not None:
            details["ctc"] = ctc
        if hidden is not None:
            details["hidden"] = hidden
        if offset is not None:  # model guidance: target-only terms (latent negatives) shift their targets too
            details["guidance_offset"] = offset
        return details
    return losses


def condition_inputs(batch):
    """Keyword arguments of the optional voice conditions a batch carries (speaker embedding, speaker context)."""
    extra = {}
    if "speaker_condition" in batch:
        extra["speaker"] = batch["speaker_condition"]
    if "context" in batch:
        extra.update(context=batch["context"], context_mask=batch["context_mask"])
    return extra


def ctc_labels(model, batch):
    """(targets [B,T], lengths [B]) of a character CTC head, () for the byte head.

    The loader normally builds them (collate); batches from elsewhere get them from their tokens here.
    """
    if getattr(getattr(model, "cfg", None), "ctc_targets", "bytes") != "chars":
        return ()
    if "ctc_targets" in batch:
        return batch["ctc_targets"], batch["ctc_target_lengths"]
    device = batch["tokens"].device
    units = getattr(model.cfg, "text_units", "bytes")
    return tuple(value.to(device) for value in char_ctc_targets(batch["tokens"].cpu(), units))


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
    extra = {"cached": cached} if cached is not None else condition_inputs(batch)
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


def ctc_alignment_loss(logits, token_valid, tokens, drop, targets=None, target_lengths=None):
    """Per-example CTC between generator frames and transcript bytes; PAD (0) is the blank.

    Examples whose text was dropped for classifier-free guidance cannot be aligned and get zero.
    Byte targets are padded [B,S] rows holding each transcript's bytes first: a stable sort of the
    "not a byte" flag compacts them on the device, where per-row boolean indexing and a Python
    list of lengths each waited for the GPU. Entries past a row's length are ignored by the loss.
    `targets` (padded [B,T] character ids, blank 0) with `target_lengths` replace the byte targets.
    """
    from .text import BYTE_OFFSET

    if targets is None:
        is_byte = tokens >= BYTE_OFFSET
        lengths = is_byte.sum(1)
        targets = tokens.gather(1, torch.sort((~is_byte).to(torch.uint8), dim=1, stable=True).indices)
    else:
        lengths = target_lengths.to(logits.device)
    loss = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets,
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


LATE_OPTION_NAMES = dict(guidance="guidance_late", rescale="cfg_rescale_late", eta="apg_eta_late",
                         norm="apg_norm_late", momentum="apg_momentum_late")


def guidance_windows(early, split, start, until, **overrides):
    """Settings of `sample`'s late guidance window: the early window's settings with the non-None `*_late` overrides.

    Without a split there is a single window and the early settings are returned unchanged; late overrides would
    then have no window to act on, so they are rejected instead of being silently ignored.
    """
    given = {k: v for k, v in overrides.items() if v is not None}
    if split is None:
        if given:
            raise ValueError(f"{sorted(LATE_OPTION_NAMES[k] for k in given)} need guidance_split")
        return early
    if not math.isfinite(split) or not start <= split <= until:
        raise ValueError("Need guidance_from <= guidance_split <= guidance_until")
    late = {**early, **given}
    if (
        not math.isfinite(late["guidance"])
        or late["guidance"] < 0
        or not 0 <= late["rescale"] <= 1
        or late["norm"] < 0
        or not -1 < late["momentum"] < 1
        or not math.isfinite(late["eta"])
    ):
        raise ValueError("Late window needs guidance >= 0, cfg_rescale in [0,1], apg_norm >= 0, apg_momentum in (-1,1)")
    return late


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
        cached=model.conditions(empty, no_prompt, tokens, segments, speaker=no_speaker(model, len(lengths))),
    )


def no_speaker(model, rows):
    """The zero speaker embedding [rows,E] of prompt-free conditions, or None for models without the condition."""
    width = getattr(model.cfg, "speaker_condition_dim", 0)
    return torch.zeros(rows, width, device=next(model.parameters()).device) if width else None


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
    guidance_split=None,
    guidance_late=None,
    apg_eta_late=None,
    apg_norm_late=None,
    apg_momentum_late=None,
    cfg_rescale_late=None,
    speaker=None,
    context=None,
    context_mask=None,
):
    """Euler sampler with classifier-free guidance.

    Guidance is applied while guidance_from <= t < guidance_until (t=0 is noise): Echo/Irodori guide only the noisy
    half of the trajectory (cfg_min_t 0.5 in their reversed convention), which keeps the alignment benefit at roughly
    half the extra forward passes. `noise_scale` shrinks the initial noise (Echo's 0.8-0.9 "truncation").
    `cfg_rescale`, `apg_eta`, `apg_norm` and `apg_momentum` reshape the guided update (see `guided_update`); their
    defaults give plain CFG. `speaker_guidance` enables independent text/speaker guidance with a third, prompt-free
    branch built by `text_only_rows` (pass it as `text_only`): v_null + g (v_text - v_null) + g_s (v_full - v_text),
    which equals plain CFG for g_s = g. It also acts where the text scale is 1 (then v_text + g_s (v_full - v_text),
    without the null branch, which cancels). It is not combined with the update shaping (rescale/APG, either window).

    `guidance_split` cuts the guided interval into an early window [guidance_from, guidance_split), which uses the
    settings above, and a late window [guidance_split, guidance_until), whose scale and update shape come from
    `guidance_late`, `cfg_rescale_late`, `apg_eta_late`, `apg_norm_late` and `apg_momentum_late` (None: the early
    value). Why: the guidance-interval study (Kynkaanniemi et al., arXiv 2404.07724) finds guidance harmful at high
    noise and redundant at low noise, and here `guidance_until 0.5` kept WER while saving 15% of the compute. APG
    (eta 0.5, momentum -0.3) and rescale 0.7 over the whole path cut the files touching the decoder's tanh ceiling
    from 96% to 58%/22% at g=5 but raised WER 4.32 -> 4.96/5.16; the alignment is settled in the noisy early steps,
    so APG/rescale confined to the late window should shape the amplitude without that WER cost (mechanistic, to be
    measured). APG momentum keeps one running average over the steps whose window uses momentum: a split without
    late overrides reproduces the single window exactly, and momentum used only late starts fresh at the split.
    """
    if steps < 1 or not -1 <= sway <= 0 or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("Invalid sampler settings")
    if not 0 < guidance_until <= 1 or not 0 < noise_scale <= 1.5 or not 0 <= guidance_from < guidance_until:
        raise ValueError("Need 0 <= guidance_from < guidance_until <= 1 and noise_scale in (0,1.5]")
    if not 0 <= cfg_rescale <= 1 or apg_norm < 0 or not -1 < apg_momentum < 1 or not math.isfinite(apg_eta):
        raise ValueError("cfg_rescale must lie in [0,1], apg_norm >= 0, apg_momentum in (-1,1)")
    if speaker_guidance is not None and (text_only is None or not math.isfinite(speaker_guidance)):
        raise ValueError("Independent speaker guidance needs the prompt-free branch (text_only_rows)")
    early = dict(guidance=guidance, rescale=cfg_rescale, eta=apg_eta, norm=apg_norm, momentum=apg_momentum)
    late = guidance_windows(early, guidance_split, guidance_from, guidance_until, guidance=guidance_late,
                            rescale=cfg_rescale_late, eta=apg_eta_late, norm=apg_norm_late, momentum=apg_momentum_late)
    if speaker_guidance is not None and any(
        (window["rescale"], window["eta"], window["norm"], window["momentum"]) != (0, 1, 0, 0)
        for window in (early, late)
    ):
        # The three-branch update does not go through guided_update: these would be reported but not applied.
        raise ValueError("speaker_guidance supports only plain CFG shaping: no cfg_rescale, apg_* or their *_late")
    split = guidance_until if guidance_split is None else guidance_split
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
        model.conditions(prompt, prompt_mask, tokens, segments, speaker=speaker, context=context,
                         context_mask=context_mask)
        if condition_cache is None
        else condition_cache
    )
    if guidance != 1 or late["guidance"] != 1:
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
    guided_steps = evaluations = late_steps = 0
    apg_state = {}
    for t0, t1 in zip(times[:-1], times[1:]):
        t = t0.expand(x.size(0))
        window = early if float(t0) < split else late
        scale = window["guidance"]
        # Speaker guidance guides even at text scale 1, where its update is v_text + g_s (v_full - v_text).
        guided = scale != 1 or speaker_guidance is not None
        if not guided or not guidance_from <= float(t0) < guidance_until:
            v = to_velocity(
                model, model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond), x, t
            )
            evaluations += 1
        else:
            if scale == 1:  # the null branch cancels (v_null + 1 (v_text - v_null) = v_text): not evaluated
                v = to_velocity(
                    model, model(x, t, prompt, prompt_mask, valid, tokens, segments, cached=cond), x, t
                )
                evaluations += 1
            else:
                both = torch.cat([x, x.masked_fill(prompt_mask[..., None], 0)])
                v, u = to_velocity(model, model(both, t.repeat(2), **pair), both, t.repeat(2)).chunk(2)
                evaluations += 2
            if speaker_guidance is None:
                v = guided_update(
                    v, u, x, t, mask, scale, window["rescale"], window["eta"], window["norm"], window["momentum"],
                    apg_state,
                )
            else:
                rows = sanitize(torch.gather(x, 1, index), text_valid)
                w = to_velocity(model, model(rows, t, valid=text_valid, **branch), rows, t)
                w = torch.zeros_like(x).scatter_add(1, index, sanitize(w, text_valid))
                v = (w if scale == 1 else u + scale * (w - u)) + speaker_guidance * (v - w)
                evaluations += 1
            guided_steps += 1
            late_steps += guidance_split is not None and float(t0) >= split
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
            guidance_split=guidance_split,
            late_window=None if guidance_split is None else dict(late, guided_steps=late_steps),
            time_grid=times.cpu().tolist(),
            solver="euler",
        )
    return (x, times, torch.stack(trajectory)) if return_trajectory else x
