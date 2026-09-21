"""Toy-scale probe: does length-aware RoPE in cross-attention (and text self-attention) make the
model use the text earlier?  Scratch subclasses only; the repository model is untouched.

Signal: held-out loss with the correct text minus loss with ANOTHER utterance's text (same batch),
tracked over training.  Zero means the text is ignored.
"""

import argparse
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dacvae_tts.config import Config
from dacvae_tts.contracts import audio_shapes, mask_values, sanitize, text_shapes
from dacvae_tts.data import BucketBatchSampler, LatentDataset, collate, move_batch
from dacvae_tts.model import Attention, Block, FlowTTS, TextEncoder, per_example_mse, sinusoidal
from dacvae_tts.text import BYTE_OFFSET

device = torch.device("cuda")
GAMMA = 10.0


def rotate(x, positions):
    """x [B,H,N,Dh], positions [B,N] (already length-normalised and scaled)."""
    half = x.size(-1) // 2
    inv_freq = torch.exp(torch.arange(half, device=x.device).float() * (-math.log(10000.0) / half))
    angles = positions[:, None, :, None].float() * inv_freq
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class RotaryCrossAttention(Attention):
    def forward(self, x, context, valid, q_pos=None, k_pos=None):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        if q_pos is not None:
            q, k = rotate(q, q_pos), rotate(k, k_pos)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class LBlock(Block):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cross_attn = RotaryCrossAttention(cfg.width, cfg.heads)

    def forward(self, x, text, valid, text_valid, cond, q_pos, k_pos):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(cond).unsqueeze(1).chunk(9, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.self_attn(h, h, valid)
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, text, text_valid, q_pos, k_pos)
        x = x + g3 * self.ff(self.norm3(x) * (1 + s3) + b3)
        return x * valid[..., None]


class AttentiveTextEncoder(TextEncoder):
    def __init__(self, cfg, layers=2):
        super().__init__(cfg)
        self.attention = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    cfg.width, cfg.heads, 2 * cfg.width, dropout=0.0, batch_first=True, norm_first=True
                )
                for _ in range(layers)
            ]
        )

    def forward(self, tokens, segments):
        valid = tokens.ne(0)
        x = self.embedding(tokens) + self.segment(segments)
        x = x + sinusoidal(torch.arange(tokens.size(1), device=x.device), x.size(-1)).to(x.dtype)
        for conv, mlp in zip(self.convs, self.mlps):
            x = x * valid[..., None]
            x = x + mlp(conv(x.transpose(1, 2)).transpose(1, 2))
        for layer in self.attention:
            x = layer(x, src_key_padding_mask=~valid)
        return self.norm(x) * valid[..., None], valid


def segment_progress(first, second):
    """Boolean [B,N] masks of two consecutive segments -> positions in [0,1) and [1,2), times GAMMA."""
    n1 = first.sum(1, keepdim=True).clamp_min(1)
    n2 = second.sum(1, keepdim=True).clamp_min(1)
    index = torch.arange(first.size(1), device=first.device)[None].float()
    position = torch.where(first, index / n1, 1 + (index - n1) / n2)
    return GAMMA * position * (first | second)


class LFlowTTS(FlowTTS):
    """patch_size must be 1 (frame positions are used directly)."""

    def __init__(self, cfg, rotary=True, text_attention=0):
        super().__init__(cfg)
        assert cfg.patch_size == 1
        self.rotary = rotary
        self.blocks = nn.ModuleList([LBlock(cfg) for _ in range(cfg.depth)])
        if text_attention:
            self.text = AttentiveTextEncoder(cfg, text_attention)

    def forward(self, x, time, prompt, prompt_mask, valid, tokens, segments, drop=None, cached=None):
        audio_shapes(x, prompt, prompt_mask, valid, self.cfg.latent_dim)
        text_shapes(tokens, segments, x.size(0), x.device)
        prompt_mask = prompt_mask & valid
        layout = prompt_mask  # positions follow the true layout even when the payload is dropped
        x, prompt = sanitize(x, valid), sanitize(prompt, prompt_mask)
        if cached is None:
            cached = self.conditions(prompt, prompt_mask, tokens, segments, drop)
        text, text_valid, voice = cached
        if drop is not None:
            x = x.masked_fill((drop[:, None] & prompt_mask)[..., None], 0)
            prompt = prompt.masked_fill(drop[:, None, None], 0)
            text = text.masked_fill(drop[:, None, None], 0)
            voice = voice.masked_fill(drop[:, None], 0)
            prompt_mask = prompt_mask & (~drop)[:, None]
        features = torch.cat([x, prompt, prompt_mask[..., None].to(x.dtype)], -1)
        h = self.input(features)
        h = h + sinusoidal(torch.arange(h.size(1), device=x.device), h.size(-1)).to(h.dtype)
        cond = self.time(sinusoidal(time * 1000, self.cfg.width).to(h.dtype)) + voice
        q_pos = k_pos = None
        if self.rotary:
            q_pos = segment_progress(layout, valid & ~layout)
            k_pos = segment_progress(text_valid & (segments == 0), text_valid & (segments == 1))
        for block in self.blocks:
            h = block(h, text, valid, text_valid, cond, q_pos, k_pos)
        return sanitize(self.output(h), valid)


VARIANTS = {
    "p1": dict(rotary=False, text_attention=0),
    "larope_p1": dict(rotary=True, text_attention=0),
    "larope_sa_p1": dict(rotary=True, text_attention=2),
}
GRID = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
MID = torch.tensor([0.25, 0.4, 0.55, 0.7])


def training_loss(model, batch, duration_weight=0.1):
    mask = mask_values(batch["valid"], batch["prompt_mask"])
    x1 = sanitize(batch["latents"], batch["valid"])
    b = x1.size(0)
    time_ = torch.rand(b, device=device)
    noise = sanitize(torch.randn_like(x1), batch["valid"])
    xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
    xt = torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]
    drop = torch.rand(b, device=device) < model.cfg.cond_dropout
    xt = xt.masked_fill((drop[:, None] & batch["prompt_mask"])[..., None], 0)
    cached = model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
    keys = ("prompt", "prompt_mask", "valid", "tokens", "segments")
    pred = model(xt, time_, *(batch[k] for k in keys), drop=drop, cached=cached)
    flow = per_example_mse(pred, x1 - noise, mask)
    frames = mask.sum(1)
    characters = ((batch["segments"] == 1) & (batch["tokens"] >= BYTE_OFFSET)).sum(1)
    duration = model.predict_duration(
        *(batch[k] for k in keys[:2]), batch["tokens"], batch["segments"], cached=cached
    )
    duration_loss = F.smooth_l1_loss(
        duration.float(), (frames / characters.clamp_min(1)).log(), reduction="none"
    )
    return (flow + duration_weight * duration_loss).mean()


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    keys = ("prompt", "prompt_mask", "valid", "tokens", "segments")
    grid_sum, grid_count = torch.zeros(len(GRID)), torch.zeros(len(GRID))
    totals = {"correct": 0.0, "shuffled_text": 0.0, "shuffled_reference": 0.0}
    count = 0
    for index, batch in enumerate(val_batches):
        generator = torch.Generator(device=device).manual_seed(1000 + index)
        mask = mask_values(batch["valid"], batch["prompt_mask"])
        x1 = sanitize(batch["latents"], batch["valid"])
        b = x1.size(0)
        noise = sanitize(torch.randn(x1.shape, device=device, generator=generator), batch["valid"])

        def state(time_):
            xt = (1 - time_[:, None, None]) * noise + time_[:, None, None] * x1
            return torch.where(batch["prompt_mask"][..., None], x1, xt) * batch["valid"][..., None]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            bins = (torch.arange(b) + index) % len(GRID)
            time_ = GRID[bins].to(device)
            error = per_example_mse(
                model(state(time_), time_, *(batch[k] for k in keys)), x1 - noise, mask
            ).cpu()
            grid_sum.scatter_add_(0, bins, error)
            grid_count.scatter_add_(0, bins, torch.ones(b))

            time_ = MID[(torch.arange(b) + index) % len(MID)].to(device)
            xt = state(time_)
            correct = model(xt, time_, *(batch[k] for k in keys))
            wrong_text = model(
                xt,
                time_,
                batch["prompt"],
                batch["prompt_mask"],
                batch["valid"],
                batch["tokens"].roll(1, 0),
                batch["segments"].roll(1, 0),
            )
            donor = batch["latents"].roll(1, 0)
            donor_length = batch["prompt_mask"].roll(1, 0).sum(1).clamp_min(1)
            positions = torch.arange(x1.size(1), device=device)[None] % donor_length[:, None]
            tiled = donor.gather(1, positions[..., None].expand_as(donor)) * batch["prompt_mask"][..., None]
            wrong_ref = model(
                torch.where(batch["prompt_mask"][..., None], tiled, xt),
                time_,
                tiled,
                batch["prompt_mask"],
                batch["valid"],
                batch["tokens"],
                batch["segments"],
            )
        for key, value in (
            ("correct", correct),
            ("shuffled_text", wrong_text),
            ("shuffled_reference", wrong_ref),
        ):
            totals[key] += float(per_example_mse(value, x1 - noise, mask).sum())
        count += b
    model.train()
    grid = grid_sum / grid_count
    result = {key: value / count for key, value in totals.items()}
    result["text_gain"] = result["shuffled_text"] - result["correct"]
    result["reference_gain"] = result["shuffled_reference"] - result["correct"]
    result["v_mse_mean"] = float(grid.mean())
    result["v_mse_by_t"] = [round(float(v), 4) for v in grid]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    train_data = LatentDataset(args.cache, "train", args.seed)
    val_data = LatentDataset(args.cache, "val", args.seed)
    val_batches = [
        move_batch(collate([val_data[(0, i)] for i in indices]), device)
        for indices in BucketBatchSampler(val_data.costs, args.batch, 0, 1, 12345).batches()[:12]
    ]
    results = {}
    for name in args.variants:
        cfg = Config.load("configs/tiny.yaml")  # run from the repository root
        cfg.model.patch_size = 1
        torch.manual_seed(args.seed)
        model = LFlowTTS(cfg.model, **VARIANTS[name]).to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01, fused=True
        )
        sampler = BucketBatchSampler(train_data.costs, args.batch, 0, 1, args.seed)
        loader = DataLoader(
            train_data,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            multiprocessing_context="spawn",
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"== {name}: params={params}", flush=True)
        history, step, epoch, started = [], 0, 0, time.time()
        torch.manual_seed(args.seed + 1)
        while step < args.steps:
            sampler.epoch = epoch
            for batch in loader:
                batch = move_batch(batch, device)
                for group in optimizer.param_groups:
                    group["lr"] = 3e-4 * min(1.0, (step + 1) / 300)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = training_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % args.eval_every == 0 or step == args.steps:
                    record = {"step": step, "epoch": epoch, "minutes": (time.time() - started) / 60}
                    record.update(evaluate(model, val_batches))
                    history.append(record)
                    print(json.dumps({"variant": name, **record}), flush=True)
                if step >= args.steps:
                    break
            epoch += 1
        results[name] = {"parameters": params, "history": history}
        json.dump(results, open(args.out, "w"), indent=1)
        del loader
    print("done")


if __name__ == "__main__":
    main()
