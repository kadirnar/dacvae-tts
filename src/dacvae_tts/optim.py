"""Muon for hidden weight matrices, AdamW for every other parameter.

Muon (Keller Jordan, https://kellerjordan.github.io/posts/muon/) replaces the momentum update of a
weight matrix by its nearest semi-orthogonal matrix, computed with a Newton-Schulz iteration. It is
only defined for hidden matrices: embeddings, boundary projections, output heads, convolution
filters, biases and normalization gains stay on AdamW, the convention of the reference
implementation and of the Echo-TTS / Irodori-TTS recipes. `torch.optim.Muon` needs PyTorch 2.9;
this module keeps the package working from 2.5 onwards and batches the iteration across equally
shaped matrices, which matters for these small, launch-bound models.

Updates are scaled by 0.2 * sqrt(max(rows, columns)) (Moonshot's RMS matching, arXiv:2502.16982) so
one learning rate and one weight decay remain meaningful for both parameter families.
"""

import math

import torch
from torch import nn

NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
# Row-wise concatenations of independent square-ish maps: orthogonalize every part separately.
FUSED_ROWS = {".kv.weight": 2, ".ada.1.weight": 9, ".ada_up.weight": 9, "ada_shared.1.weight": 9}
# Boundary layers (raw latents in, velocities / log-rate out) follow the AdamW convention.
BOUNDARY = {
    "input.weight",
    "ref.0.weight",
    "ref.input.weight",
    "output.1.weight",
    "duration.2.weight",
    "ctc.weight",
}


def orthogonalize(matrices, steps=5):
    """Batched Newton-Schulz on [N,rows,cols]; singular values are driven towards one."""
    if matrices.ndim != 3:
        raise ValueError("orthogonalize expects [N,rows,cols]")
    x = matrices.to(torch.bfloat16 if matrices.is_cuda else torch.float32)
    wide = x.size(-2) <= x.size(-1)
    if not wide:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NEWTON_SCHULZ
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x if wide else x.mT).to(matrices.dtype)


def muon_parts(name, parameter, embeddings=frozenset()):
    """Number of independently orthogonalized row blocks; 0 keeps the parameter on AdamW."""
    if parameter.ndim != 2 or min(parameter.shape) < 2 or id(parameter) in embeddings or name in BOUNDARY:
        return 0
    for suffix, parts in FUSED_ROWS.items():
        if name.endswith(suffix) and parameter.size(0) % parts == 0:
            return parts
    return 1


def partition(model):
    """Split trainable parameters into (muon parameters, their part counts, AdamW parameters)."""
    embeddings = frozenset(
        id(p) for module in model.modules() if isinstance(module, nn.Embedding) for p in module.parameters()
    )
    matrices, parts, others = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = muon_parts(name, parameter, embeddings)
        if count:
            matrices.append(parameter)
            parts.append(count)
        else:
            others.append(parameter)
    return matrices, parts, others


class Muon(torch.optim.Optimizer):
    """Single optimizer holding a Muon group and an AdamW group; standard `state_dict` semantics."""

    def __init__(
        self,
        matrices,
        parts,
        others,
        lr=3e-4,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        steps=5,
        betas=(0.9, 0.95),
        eps=1e-8,
    ):
        if len(matrices) != len(parts) or any(p.ndim != 2 for p in matrices):
            raise ValueError("Muon parameters must be matrices with one part count each")
        if lr < 0 or weight_decay < 0 or not 0 <= momentum < 1 or steps < 1:
            raise ValueError("Invalid Muon hyperparameters")
        groups = []
        if matrices:
            groups.append(dict(params=matrices, muon=True, parts=list(parts), momentum=momentum))
        if others:
            groups.append(dict(params=others, muon=False, betas=tuple(betas), eps=eps))
        if not groups:
            raise ValueError("No trainable parameters")
        super().__init__(groups, dict(lr=lr, weight_decay=weight_decay, nesterov=nesterov, steps=steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            (self._muon if group["muon"] else self._adamw)(group)
        return loss

    def _muon(self, group):
        lr, decay, momentum = group["lr"], group["weight_decay"], group["momentum"]
        buckets = {}
        for parameter, parts in zip(group["params"], group["parts"], strict=True):
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer = state["momentum_buffer"].lerp_(parameter.grad, 1 - momentum)
            update = parameter.grad.lerp(buffer, momentum) if group["nesterov"] else buffer
            blocks = update.reshape(parts, -1, update.size(1))
            buckets.setdefault(tuple(blocks.shape[1:]), []).append((parameter, blocks))
        # One batched iteration per distinct block shape instead of one per matrix.
        for (rows, columns), items in buckets.items():
            stacked = torch.cat([blocks for _, blocks in items])
            stacked = orthogonalize(stacked, group["steps"]) * (0.2 * math.sqrt(max(rows, columns)))
            offset = 0
            for parameter, blocks in items:
                block = stacked[offset : offset + len(blocks)]
                offset += len(blocks)
                parameter.mul_(1 - lr * decay).add_(block.reshape_as(parameter), alpha=-lr)

    def _adamw(self, group):
        lr, decay, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        parameters, gradients, first, second, step_sizes, corrections = [], [], [], [], [], []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if not state:
                state.update(
                    step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter)
                )
            state["step"] += 1
            parameters.append(parameter)
            gradients.append(parameter.grad)
            first.append(state["exp_avg"])
            second.append(state["exp_avg_sq"])
            step_sizes.append(-lr / (1 - beta1 ** state["step"]))
            corrections.append(math.sqrt(1 - beta2 ** state["step"]))
        if not parameters:
            return
        torch._foreach_mul_(parameters, 1 - lr * decay)
        torch._foreach_lerp_(first, gradients, 1 - beta1)
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, gradients, gradients, 1 - beta2)
        denominators = torch._foreach_sqrt(second)
        torch._foreach_div_(denominators, corrections)
        torch._foreach_add_(denominators, eps)
        torch._foreach_addcdiv_(parameters, first, denominators, step_sizes)


def build_optimizer(model, name, lr, weight_decay, momentum=0.95, fused=False):
    """`muon` (default recipe) or the previous plain `adamw`, selected by configuration."""
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), fused=fused
        )
    if name != "muon":
        raise ValueError("optimizer must be muon or adamw")
    matrices, parts, others = partition(model)
    return Muon(matrices, parts, others, lr=lr, weight_decay=weight_decay, momentum=momentum)
