"""Opt-in diagnostics; no extra random draws and no parameter updates."""

import math

import torch


def gradient_contributions(model, flow, duration):
    params = [p for p in model.parameters() if p.requires_grad]
    result = {}
    for name, loss in (("flow", flow), ("weighted_duration", duration)):
        if not loss.requires_grad:  # e.g. rule-based duration: no learned head, no gradient
            result[f"{name}_gradient_norm"] = 0.0
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        norm = sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt()
        result[f"{name}_gradient_norm"] = float(norm)
    return result


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split(".")[0]
            squared = parameter.grad.detach().float().square().sum()
            groups[group] = groups.get(group, 0) + squared
    return {key: float(value.sqrt()) for key, value in groups.items()}


class ActivationProbe:
    def __init__(self, model):
        self.handles, self.values = [], {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):

                def hook(_, inputs, output, key=name):
                    self.values[key] = output.detach().float().abs().amax()

                self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        result = {key: float(value) for key, value in self.values.items()}
        if any(not math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Nonfinite activation detected")
        return result


def loss_buckets(flow, times, frames):
    """Five time intervals and four target-length intervals; per-utterance means."""
    stats = torch.zeros(9, 2, device=flow.device)
    time_bin = (times.detach() * 5).long().clamp(0, 4)
    length_bin = torch.bucketize(frames.detach(), frames.new_tensor([100, 250, 500])) + 5
    for index in (time_bin, length_bin):
        stats[:, 0].scatter_add_(0, index, flow.detach().float())
        stats[:, 1].scatter_add_(0, index, torch.ones_like(flow, dtype=torch.float32))
    return stats
