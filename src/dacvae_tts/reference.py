"""Optional from-scratch reference encoder/pooling ablations; baseline stays in model.py."""

import torch
from torch import nn

from .contracts import sanitize


class TemporalReference(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.input = nn.Linear(channels, width)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2, groups=width) for _ in range(2)])
        self.mlps = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()) for _ in range(2)]
        )

    def forward(self, x, mask):
        x = sanitize(self.input(sanitize(x, mask)), mask)
        for conv, mlp in zip(self.convs, self.mlps):
            x = sanitize(x + mlp(conv(x.transpose(1, 2)).transpose(1, 2)), mask)
        return x


class ReferencePool(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode = mode
        self.projection = nn.Linear(width * 2, width) if mode == "mean_std" else None
        self.attention = nn.Linear(width, 1) if mode == "attention" else None

    def forward(self, x, mask):
        x = sanitize(x, mask)
        count = mask.sum(1, keepdim=True).clamp_min(1)
        mean = x.sum(1) / count
        if self.mode == "mean_std":
            var = sanitize((x - mean[:, None]).square(), mask).sum(1) / count
            result = self.projection(torch.cat([mean, var.clamp_min(1e-8).sqrt()], -1))
        elif self.mode == "attention":
            logits = self.attention(x).squeeze(-1).float().masked_fill(~mask, -1e9)
            weights = logits.softmax(-1) * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            result = (x * weights[..., None]).sum(1)
        else:
            result = mean
        return result.masked_fill(~mask.any(1, keepdim=True), 0)
